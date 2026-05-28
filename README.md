# HLEWM — Hierarchical LeWorldModel

This codebase implements two things built on top of each other:

1. **LeWM (flat)** — a JEPA world model trained end-to-end from pixels. Plans actions via CEM in a 192-dim latent space.
2. **HLEWM (hierarchical)** — adds a second-level world model on top of a frozen LeWM. The high level plans *where to go* (subgoal embeddings); the low level plans *how to get there* (primitive actions).

You always train L1 first, then optionally train L2 on top of a frozen L1 checkpoint.

Supported environments: **PushT**, **OGBench Cube**, **OGBench Scene**.

---

## Installation

```bash
conda activate hlewm
```

The `hlewm` conda environment already has all dependencies. If starting fresh:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

---

## Data

Training and evaluation both require offline expert demonstration datasets in HDF5 format.
All datasets live under `$STABLEWM_HOME/datasets/` (defaults to `~/.stable_worldmodel/datasets/`).

| File | Environment |
|---|---|
| `pusht_expert_train.h5` | PushT |
| `ogbench--cube_single_expert.h5` | OGBench Cube |
| `ogbench--scene_single_expert.h5` | OGBench Scene |

The `--` in the OGBench filenames is not a typo — it is how swm caches downloaded datasets (replacing `/` with `--`).

### PushT

Download from HuggingFace and place in the datasets folder:

```bash
hf download quentinll/lewm-pusht --repo-type dataset --local-dir /tmp/pusht
zstd -d /tmp/pusht/pusht_expert_train.h5.zst -o ~/.stable_worldmodel/datasets/pusht_expert_train.h5
```

### OGBench (Cube and Scene)

OGBench datasets are not on HuggingFace. You have two options:

**Option A — get the pre-built files from a collaborator.**
The files need to land at their exact names in `~/.stable_worldmodel/datasets/`. If you receive them as `.tar.zst` archives, extract them:

```bash
tar --zstd -xvf cube_single_expert.tar.zst -C ~/.stable_worldmodel/datasets/
# rename to match expected filename if necessary:
mv ~/.stable_worldmodel/datasets/cube_single_expert.h5 \
   ~/.stable_worldmodel/datasets/ogbench--cube_single_expert.h5
```

**Option B — collect from scratch using the expert policy.**
swm ships an oracle policy for OGBench. This takes ~30 min per 1000 episodes on a single CPU:

```bash
python collect_ogbench.py --env cube --episodes 1000
python collect_ogbench.py --env scene --episodes 1000
```

### Verify your setup

```bash
ls ~/.stable_worldmodel/datasets/
# pusht_expert_train.h5  ogbench--cube_single_expert.h5  ogbench--scene_single_expert.h5
```

---

## Training

### Step 1 — Train L1 (flat LeWM)

This trains the core world model: a ViT encoder + autoregressive predictor, regularised by SIGReg to prevent collapse.

```bash
# PushT
python train.py data=pusht

# OGBench Cube
python train.py data=ogb

# OGBench Scene
python train.py data=ogb_scene
```

The config file is `config/train/lewm.yaml`. The `data=` argument swaps only the dataset config (under `config/train/data/`); everything else (model, optimizer, trainer) stays the same.

Checkpoints are saved to `$STABLEWM_HOME/checkpoints/<run_id>/` as `weights.pt` + `config.json`.

### Step 2 — Train L2 (hierarchical top level, optional)

L2 learns to predict *waypoints* in the L1 latent space, conditioned on macro-action embeddings. It requires a trained L1 checkpoint.

```bash
# PushT
python train.py --config-name hlewm data=pusht_l2 l2.l1_checkpoint=<L1_run_id>

# OGBench Cube
python train.py --config-name hlewm data=ogb_l2 l2.l1_checkpoint=<L1_run_id>

# OGBench Scene
python train.py --config-name hlewm data=ogb_scene_l2 l2.l1_checkpoint=<L1_run_id>
```

`<L1_run_id>` is the folder name under `$STABLEWM_HOME/checkpoints/` (e.g. `1` if Hydra gave your job ID 1), or a HuggingFace repo ID (e.g. `FadyRezk/lewm-pusht-fixed`).

The `pusht_l2` / `ogb_l2` / `ogb_scene_l2` configs differ from their L1 counterparts only in `num_steps` (longer sequences, needed for waypoint sampling).

---

## Evaluation

### Flat L1 eval

Uses `eval.py`. The policy runs CEM in the 192-dim latent space to plan primitive actions.

```bash
# PushT
python eval.py --config-name pusht policy=<checkpoint>

# OGBench Cube
python eval.py --config-name cube policy=<checkpoint>

# OGBench Scene
python eval.py --config-name scene policy=<checkpoint>
```

`<checkpoint>` is a run folder under `$STABLEWM_HOME/checkpoints/`, or a HuggingFace repo ID.

### Hierarchical L2+L1 eval

Uses `heval.py`. L2 CEM plans a macro-action embedding sequence → extracts first subgoal → L1 CEM reaches it with primitive actions.

```bash
# PushT
python heval.py --config-name hpusht policy=<HJEPA_checkpoint>

# OGBench Cube
python heval.py --config-name hcube policy=<HJEPA_checkpoint>

# OGBench Scene
python heval.py --config-name hscene policy=<HJEPA_checkpoint>
```

The HJEPA checkpoint is what L2 training produces — it bundles the frozen L1 and the trained L2 predictor together.

### Config map

| Script | Config name | Environment |
|---|---|---|
| `eval.py` | `pusht` | PushT (flat) |
| `eval.py` | `cube` | OGBench Cube (flat) |
| `eval.py` | `scene` | OGBench Scene (flat) |
| `heval.py` | `hpusht` | PushT (hierarchical) |
| `heval.py` | `hcube` | OGBench Cube (hierarchical) |
| `heval.py` | `hscene` | OGBench Scene (hierarchical) |

---

## Pretrained checkpoints

L1 checkpoints are on HuggingFace. Pass the repo ID as `policy=` directly — swm downloads and caches them automatically.

```bash
python eval.py --config-name pusht policy=FadyRezk/lewm-pusht-fixed
```

Expected: **86%** success on PushT.

---

## Quick-start smoke test

To verify the hierarchical pipeline without running full training, bootstrap a random HJEPA checkpoint and run it:

```bash
python bootstrap_hjepa.py        # creates a random-weight HJEPA under $STABLEWM_HOME/checkpoints/hlewm/
python heval.py --config-name hpusht policy=hlewm
```

This will score near 0% but confirms the full stack (data loading, model, CEM, env) is wired correctly.
