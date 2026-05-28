"""LeWM model builder.

Provides patch_cfg and build with the standard interface expected by train.py:
    patch_cfg(cfg, train_level, raw_action_dim)  -- mutates cfg in-place
    build(cfg, train_level) -> (model, forward_fn, module_kwargs, ckpt_cfg)
"""
from pathlib import Path

import hydra
import stable_worldmodel as swm
from omegaconf import OmegaConf

from models.lewm.wm import JEPA
from models.lewm.hier import HJEPA
from models.modules.module import SIGReg


def patch_cfg(cfg, train_level, raw_action_dim):
    cfg.model.action_encoder.input_dim = raw_action_dim
    if train_level == 2:
        cfg.l2.model.macro_action_encoder.action_dim = raw_action_dim


def _build_l1(cfg):
    world_model = hydra.utils.instantiate(cfg.model)
    forward_fn = JEPA.training_forward
    module_kwargs = {"sigreg": SIGReg(**cfg.loss.sigreg.kwargs)}
    ckpt_cfg = cfg.model
    return world_model, forward_fn, module_kwargs, ckpt_cfg


def _build_l2(cfg):
    l1_jepa = swm.wm.utils.load_pretrained(cfg.l2.l1_checkpoint)
    macro_action_encoder = hydra.utils.instantiate(cfg.l2.model.macro_action_encoder)
    l2_predictor = hydra.utils.instantiate(cfg.l2.model.predictor)
    l2_pred_proj = hydra.utils.instantiate(cfg.l2.model.pred_proj)
    world_model = HJEPA(
        l1_jepa=l1_jepa,
        macro_action_encoder=macro_action_encoder,
        l2_predictor=l2_predictor,
        l2_pred_proj=l2_pred_proj,
    )

    cache_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    _, l1_cfg = swm.wm.utils._resolve(str(cfg.l2.l1_checkpoint), cache_dir)
    l1_cfg["_target_"] = "models.lewm.wm.JEPA"
    l2 = OmegaConf.to_container(cfg.l2.model, resolve=True)
    ckpt_cfg = OmegaConf.create({
        "_target_": "models.lewm.hier.HJEPA",
        "l1_jepa": l1_cfg,
        "macro_action_encoder": l2["macro_action_encoder"],
        "l2_predictor": l2["predictor"],
        "l2_pred_proj": l2["pred_proj"],
    })

    return world_model, HJEPA.training_forward, {}, ckpt_cfg


def build(cfg, train_level):
    if train_level == 1:
        return _build_l1(cfg)
    return _build_l2(cfg)
