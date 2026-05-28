import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from hjepa import HJEPA
from jepa import JEPA
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def _build_hjepa_config(cfg):
    """Return an OmegaConf DictConfig that load_pretrained can use to reconstruct HJEPA."""
    cache_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
    _, l1_cfg = swm.wm.utils._resolve(str(cfg.l2.l1_checkpoint), cache_dir)
    l1_cfg["_target_"] = "jepa.JEPA"
    l2 = OmegaConf.to_container(cfg.l2.model, resolve=True)
    return OmegaConf.create({
        "_target_": "hjepa.HJEPA",
        "l1_jepa": l1_cfg,
        "macro_action_encoder": l2["macro_action_encoder"],
        "l2_predictor": l2["predictor"],
        "l2_pred_proj": l2["pred_proj"],
    })


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
            cfg=cfg,
            forward=JEPA.training_forward,
            optim=optimizers,
        )
        ckpt_cfg = cfg.model
    else:
        world_model = _build_hjepa(cfg)
        module = spt.Module(
            model=world_model,
            cfg=cfg,
            forward=HJEPA.training_forward,
            optim=optimizers,
        )
        ckpt_cfg = _build_hjepa_config(cfg)

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
