"""Bootstrap an HJEPA checkpoint from an existing L1 (LeWM) checkpoint.

L2 weights are randomly initialized — results are meaningless but the pipeline
runs end-to-end, which is the point.

Usage:
    python bootstrap_hjepa.py
Then evaluate:
    python heval.py --config-name hpusht policy=hlewm
"""
import json
import torch
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.wm.utils import _resolve

from models.lewm.hier import HJEPA
from models.modules.module import MacroActionEncoder, ARPredictor, MLP

L1_NAME = "FadyRezk/lewm-pusht-fixed"
OUT_NAME = "hlewm"
EMBED_DIM = 192
ACTION_DIM = 10   # frameskip(5) * raw_action_dim(2) for PushT
NUM_WAYPOINTS = 6

cache_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))

_, l1_config = _resolve(L1_NAME, cache_dir)
l1_config["_target_"] = "models.lewm.wm.JEPA"
l1_jepa = swm.wm.utils.load_pretrained(L1_NAME)

macro_action_encoder = MacroActionEncoder(
    action_dim=ACTION_DIM, macro_dim=EMBED_DIM, hidden_dim=EMBED_DIM,
    depth=4, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1,
)
l2_predictor = ARPredictor(
    num_frames=NUM_WAYPOINTS, input_dim=EMBED_DIM, hidden_dim=EMBED_DIM,
    output_dim=EMBED_DIM, depth=6, heads=16, mlp_dim=2048, dim_head=64,
    dropout=0.1, emb_dropout=0.0,
)
l2_pred_proj = MLP(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, hidden_dim=2048,
    norm_fn=torch.nn.BatchNorm1d,
)

hjepa = HJEPA(
    l1_jepa=l1_jepa,
    macro_action_encoder=macro_action_encoder,
    l2_predictor=l2_predictor,
    l2_pred_proj=l2_pred_proj,
)

hjepa_config = {
    "_target_": "models.lewm.hier.HJEPA",
    "l1_jepa": l1_config,
    "macro_action_encoder": {
        "_target_": "models.modules.module.MacroActionEncoder",
        "action_dim": ACTION_DIM, "macro_dim": EMBED_DIM, "hidden_dim": EMBED_DIM,
        "depth": 4, "heads": 8, "dim_head": 64, "mlp_dim": 1024, "dropout": 0.1,
    },
    "l2_predictor": {
        "_target_": "models.modules.module.ARPredictor",
        "num_frames": NUM_WAYPOINTS, "input_dim": EMBED_DIM, "hidden_dim": EMBED_DIM,
        "output_dim": EMBED_DIM, "depth": 6, "heads": 16, "mlp_dim": 2048,
        "dim_head": 64, "dropout": 0.1, "emb_dropout": 0.0,
    },
    "l2_pred_proj": {
        "_target_": "models.modules.module.MLP",
        "input_dim": EMBED_DIM, "output_dim": EMBED_DIM, "hidden_dim": 2048,
        "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
    },
}

out_dir = cache_dir / OUT_NAME
out_dir.mkdir(exist_ok=True)
torch.save(hjepa.state_dict(), out_dir / "weights.pt")
(out_dir / "config.json").write_text(json.dumps(hjepa_config, indent=2))
print(f"Saved bootstrap HJEPA → {out_dir}")
