import os
import random
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from hjepa import HJEPA
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """L1 flat LeWM forward: encode → predict → MSE + SIGReg."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]      # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]           # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def l2_forward(self, batch, stage, cfg):
    """L2 high-level world model forward: waypoint MSE only, no SIGReg.

    Randomly samples N waypoint indices (shared across the batch), extracts
    variable-length action chunks between consecutive waypoints, encodes with
    the frozen L1 encoder, then trains the L2 predictor to predict each next
    waypoint embedding from the current one and its macro-action.
    """
    T = batch["pixels"].size(1)
    N = cfg.l2.num_waypoints
    min_gap = cfg.l2.min_waypoint_gap

    # Sample shared waypoint indices: fixed endpoints, random intermediates.
    # 'shared across batch' means no padding is needed — all elements in this
    # batch have the same chunk lengths.
    pool = list(range(min_gap, T - min_gap))
    intermediates = sorted(random.sample(pool, N - 2))
    wp_indices = [0] + intermediates + [T - 1]

    # Waypoint pixels: (B, N, C, H, W)
    pixels_wp = batch["pixels"][:, wp_indices]

    # Variable-length action chunks between consecutive waypoints.
    # chunk[k]: (B, L_k, action_dim)  where L_k = wp_indices[k+1] - wp_indices[k]
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    action_chunks = [
        batch["action"][:, wp_indices[k]:wp_indices[k + 1]]
        for k in range(N - 1)
    ]

    # Encode waypoints with frozen L1 encoder → (B, N, D)
    wp_embs = self.model.encode_waypoints(pixels_wp)

    # Encode macro-actions → (B, N-1, D)
    macro_embs = self.model.encode_macro_actions(action_chunks)

    # L2 teacher-forced prediction → (B, N-1, D)
    pred_embs = self.model.predict(wp_embs, macro_embs)
    tgt_embs = wp_embs[:, 1:].detach()   # targets from frozen encoder; detach is redundant but explicit

    loss = (pred_embs - tgt_embs).pow(2).mean()

    self.log(f"{stage}/l2_pred_loss", loss.detach(), on_step=True, sync_dist=True)
    return {"loss": loss}


def _build_hjepa(cfg):
    """Load frozen L1 JEPA from checkpoint and build HJEPA with fresh L2 components."""
    l1_jepa = swm.wm.utils.load_pretrained(cfg.l2.l1_checkpoint)

    macro_action_encoder = hydra.utils.instantiate(cfg.l2.model.macro_action_encoder)
    l2_predictor = hydra.utils.instantiate(cfg.l2.model.predictor)
    l2_pred_proj = hydra.utils.instantiate(cfg.l2.model.pred_proj)

    return HJEPA(
        l1_jepa=l1_jepa,
        macro_action_encoder=macro_action_encoder,
        l2_predictor=l2_predictor,
        l2_pred_proj=l2_pred_proj,
    )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    train_level = cfg.get("train_level", 1)

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    raw_action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        if train_level == 1:
            cfg.model.action_encoder.input_dim = raw_action_dim
        else:
            # L1 model config is present for reference; set L2 macro action encoder dim.
            cfg.model.action_encoder.input_dim = raw_action_dim  # kept for completeness
            cfg.l2.model.macro_action_encoder.action_dim = raw_action_dim

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    if train_level == 1:
        world_model = hydra.utils.instantiate(cfg.model)
        module = spt.Module(
            model=world_model,
            sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
            forward=partial(lejepa_forward, cfg=cfg),
            optim=optimizers,
        )
        ckpt_cfg = cfg.model
    else:
        world_model = _build_hjepa(cfg)
        module = spt.Module(
            model=world_model,
            forward=partial(l2_forward, cfg=cfg),
            optim=optimizers,
        )
        ckpt_cfg = cfg.l2.model

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=ckpt_cfg, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
