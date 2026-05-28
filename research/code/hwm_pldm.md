# HWM_PLDM Codebase Reference

**Scope:** This repo implements the **Diverse Maze (PLDM backend)** experiments from the HWM paper. The Franka (VJEPA2-AC backend) and Push-T (DINO-WM backend) experiments use separate codebases not present here. Many paper claims about transformers, CLS tokens, and L1 loss refer to those backends; the maze backend here diverges in important ways documented below.

---

## 1. Layout

```
HWM_PLDM/
├── pldm/                          # All training, modeling, and planning code
│   ├── train.py                   # Main entry point: training + eval loop (737 lines)
│   ├── configs.py                 # ConfigBase dataclass + OmegaConf integration
│   ├── logger.py                  # Logging, W&B integration
│   ├── utils.py                   # General utilities
│   ├── download_ckpt_from_hf.py  # Downloads pretrained checkpoints from HuggingFace
│   ├── models/
│   │   ├── hjepa.py               # HJEPA: two-level hierarchical wrapper (211 lines)
│   │   ├── jepa.py                # JEPA: single-level world model (269 lines)
│   │   ├── decoders.py            # Output decoders (reconstruction only)
│   │   ├── misc.py                # Auxiliary modules
│   │   ├── encoders/              # Backbone encoder architectures
│   │   │   └── (ResNet, IMPALA, MENets variants)
│   │   └── predictors/
│   │       ├── sequence_predictor.py   # Base recurrent/transformer predictor
│   │       └── conv_predictors.py      # Convolutional predictor (primary for maze)
│   ├── data/
│   │   ├── dataset_factory.py     # Creates train/val/probe DataLoaders
│   │   ├── enums.py               # Dataset type enums
│   │   └── utils.py               # Data utilities
│   ├── objectives/
│   │   ├── prediction.py          # MSE prediction loss (teacher-forcing)
│   │   ├── kl.py                  # KL divergence on latent actions z
│   │   ├── idm.py                 # Inverse dynamics model loss
│   │   ├── vicreg.py              # VICReg representation learning loss
│   │   └── probe.py               # Probing losses
│   ├── planning/
│   │   ├── planners/
│   │   │   ├── mppi_planner.py    # MPPI planner (used for maze)
│   │   │   └── two_lvl_planner.py # Hierarchical two-level wrapper
│   │   ├── d4rl/
│   │   │   ├── hmpc.py            # Hierarchical MPC eval harness
│   │   │   └── mpc.py             # Flat MPC eval harness
│   │   ├── objectives.py / objectives_v2.py  # Planning cost functions
│   │   └── utils.py, plotting.py
│   ├── evaluation/
│   │   └── evaluator.py           # Top-level eval dispatcher
│   ├── probing/
│   │   └── evaluator.py           # Probing harness (linear/MLP probes)
│   ├── optimizers/                # Adam/LARS + schedulers
│   └── configs/
│       └── diverse_maze/icml/
│           ├── large_diverse_25maps.yaml      # L1 (flat PLDM) training config
│           └── large_diverse_25maps_l2.yaml   # L2 (HWM high-level) training config
├── pldm_envs/                     # Environment + dataset code
│   ├── diverse_maze/
│   │   ├── d4rl.py                # D4RLDataset: loads offline trajectories from disk
│   │   ├── enums.py               # D4RLSample NamedTuple, DatasetConfig
│   │   ├── evaluation/            # Maze-specific planning evaluator
│   │   ├── data_generation/       # Scripts to generate maze datasets
│   │   ├── transforms.py / wrappers.py
│   │   └── maze_draw.py, utils.py
│   └── utils/
│       ├── normalizer.py          # State/action normalization + L2 latent bounds
│       └── distributions.py       # Action distribution classes
└── scripts/
    ├── random_search_wall.py      # Hyperparameter search launcher (maze)
    └── random_search_mani.py      # Hyperparameter search (manipulation)
```

---

## 2. Entry Points

**Training configs:** Everything is driven by YAML + OmegaConf. CLI overrides via `--values key=value`.

```bash
# Download pretrained checkpoints
cd HWM_PLDM
python pldm/download_ckpt_from_hf.py --out-dir ./pldm/pretrained

# Train low-level (L1) PLDM world model from scratch
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps.yaml

# Train high-level (L2 / HWM) world model, loading pretrained L1
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values load_checkpoint_path=./pldm/pretrained/3-9-1-seed248_epoch=3_sample_step=15465472.ckpt

# Eval flat (L1-only) planning
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps.yaml \
  --values eval_only=true load_checkpoint_path=<l1_ckpt>

# Eval hierarchical (L2 + L1) planning
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values eval_only=true load_checkpoint_path=<l2_ckpt>

# Quick debug mode (short run)
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps.yaml \
  --values quick_debug=true
```

**Required dataset files** (paths set in YAML, must exist on disk):
- Images: `$root_path/data/maze2d_large_diverse_25maps/images.npy`
- Trajectories: `$root_path/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/data.p`
- Eval starts/targets: `$root_path/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe/starts_targets_*.pt`

`root_path` is currently hardcoded in the YAML as `/scratch/wz1232/HWM_PLDM` — **must be overridden** for any other system via `--values root_path=/your/path`.

---

## 3. Forward + Backward Trace

### 3.1 Low-Level (L1) Training

**Data loading** (`pldm/data/dataset_factory.py:53–140`, class `D4RLDataset` at `pldm_envs/diverse_maze/d4rl.py:17`):
- Loads `images.npy` (pre-rendered observations) and `data.p` (actions, positions, velocities).
- Returns `D4RLSample` namedtuples with tensors of shape `(T, B, ...)`.
- L1 trajectory length: `n_steps=15`. Each sample is 15 consecutive timesteps.

**Forward pass** (`pldm/train.py:494` → `pldm/models/hjepa.py:140–156` → `pldm/models/jepa.py`):

```
input_states: (T=15, B, C=3, H=98, W=98)   # raw image observations
actions:      (T-1=14, B, A=2)               # 2D continuous actions

level1.forward_posterior(input_states, actions)
  → level1.backbone.forward_multiple(input_states)
      # MENets backbone, subclass d4rl_a
      → encodings: (T, B, C'=32, H'=35, W'=35)   # spatial feature maps

  → level1.predictor.forward_multiple(encodings, actions, T=15)
      # ConvPredictor, subclass d4rl_b_p
      # Teacher-forcing: each step sees ground-truth encoding as input
      → predictions: (T, B, C', H', W')
```

**Loss** (`pldm/objectives/prediction.py:83`):
```python
pred_loss = (encodings[1:] - predictions[1:]).pow(2).mean()
# MSE between ground-truth encodings and predictions, shifted by 1
```

**Additional L1 losses** (from objectives_l1 config):
- VICReg: variance + invariance + covariance on encodings (std_coeff=29.4, cov_coeff=17.9)
- IDM: inverse dynamics model, predicts action from consecutive encoding pair
- PredictionProprio: MSE on proprioceptive prediction (coeff=2.42)

**Backward:** `total_loss.backward()` updates all of `level1.backbone` and `level1.predictor`. No EMA. No stop-gradient.

---

### 3.2 High-Level (L2) Training

**Data loading — L2 branch** (`pldm_envs/diverse_maze/d4rl.py:227–240`):
- Trajectories subsampled with `l2_step_skip=10`: every 10th frame becomes a waypoint.
- `l2_n_steps=6`: 6 waypoints per L2 sample, spanning 61 L1 timesteps.
- Actions grouped into chunks of size `l2_step_skip`: each L2 action is 10 L1 actions.
- Chunked actions shape: `(n_chunks=6, B, chunk_size=10, A=2)`.

**L2 action encoding** (`pldm/models/hjepa.py:79–104`):
```python
def encode_actions(self, actions):
    # actions: (T=60, B, 2)  — 6 chunks × 10 steps each
    actions = actions.view(T // step_skip, step_skip, B, 2)  # (6, 10, B, 2)
    l2_actions = actions.sum(dim=1)                            # (6, B, 2)
    l2_actions = normalizer.normalize_l2_action(l2_actions)   # normalize
    return l2_actions
```
**This is a fixed sum, not a learned encoder.** No neural network parameters. `correct_l2_merge=False` by default (normalizes before sum when True).

**L2 predictor — latent action** (`pldm/models/predictors/sequence_predictor.py`):
The L2 predictor has `z_dim=8` (from config `level2.predictor.z_dim`). At each L2 timestep, a stochastic latent variable `z ∈ R^8` is produced by a learned posterior model conditioned on the summed action chunk. This `z` is concatenated with the summed actions as input to the predictor. The posterior architecture is `'32-32'` (MLP: action → 32 → 32 → 8D mean+var). **This stochastic z is the actual "latent action" in the code — not the summation output.**

**L2 forward pass** (`pldm/models/hjepa.py:162–187`):
```
l2_states: (T=6, B, C=3, H=98, W=98)   # waypoint observations (every 10th frame)

# Step 1: Encode waypoint observations through frozen L1 backbone
level1.forward_posterior(l2_states, actions=None, encode_only=True)
  → l1_encodings: (6, B, 32, 35, 35)

# Step 2: Run L2 predictor on L1 latent states + summed actions
level2.forward_posterior(l1_encodings, actions=l2_actions)
  → predictions: (6, B, 32, 35, 35)
```

**L2 loss** (`pldm/objectives/prediction.py:83`):
```python
pred_loss = (l1_encodings[1:] - l2_predictions[1:]).pow(2).mean()
# MSE between L1 latent encodings at waypoints and L2 predictions
```

**Backward:** Updates `level2` only. `level1` is frozen (`freeze_l1=True` in config → `requires_grad=False`).

**KL loss** (`pldm/objectives/kl.py:116–143`): If `z_dim > 0`, penalizes KL(posterior || prior). For maze, prior is a standard Gaussian; posterior is the MLP conditioned on the chunked actions.

---

## 4. Module Map

| Component | File:Line | Role | Interface |
|---|---|---|---|
| `HJEPA` | `models/hjepa.py:28` | Two-level hierarchical wrapper | `config: HJEPAConfig`, `input_dim`, `normalizer` |
| `HJEPA.encode_actions` | `models/hjepa.py:79` | Sum + normalize L1 action chunks → L2 actions | `actions: (T, B, A)` → `(T//skip, B, A)` |
| `HJEPA.forward_posterior` | `models/hjepa.py:122` | L1 training pass OR L2 encoding + prediction pass | optional L1 and L2 inputs → `ForwardResult(l1, l2)` |
| `JEPA` | `models/jepa.py:39` | Single-level JEPA: backbone + predictor | `config: JEPAConfig`, `l2: bool` |
| `JEPA.forward_posterior` | `models/jepa.py:~80` | Full forward with ground-truth states (teacher-forcing) | `input_states, actions` → `JEPAForwardResult` |
| `JEPA.forward_prior` | `models/jepa.py:~120` | Autoregressive rollout (inference only) | `input_states, actions` → `JEPAForwardResult` |
| `D4RLDataset` | `pldm_envs/diverse_maze/d4rl.py:17` | Loads offline trajectory data from disk | `D4RLDatasetConfig` → `D4RLSample` namedtuples |
| `ConvPredictor` | `models/predictors/conv_predictors.py:237` | Convolutional world model predictor (primary) | spatial latent + action → next spatial latent |
| `SequencePredictor` | `models/predictors/sequence_predictor.py:17` | Base class with teacher-forcing + rollout logic | handles z_dim, posterior, prior |
| `PredictionObjective` | `objectives/prediction.py:39` | MSE prediction loss | `ForwardResult` → `LossInfo(total_loss)` |
| `KLObjective` | `objectives/kl.py:116` | KL(posterior \|\| prior) on latent z | `ForwardResult` → `LossInfo` |
| `IDMObjective` | `objectives/idm.py` | Inverse dynamics: predict action from z_t, z_{t+1} | `ForwardResult` → `LossInfo` |
| `VICRegObjective` | `objectives/vicreg.py` | VICReg collapse prevention | `encodings` → `LossInfo` |
| `Normalizer` | `pldm_envs/utils/normalizer.py:48` | Normalize/unnormalize states, actions, L2 latents | static methods; constructed from dataset stats |
| `MPPIPlanner` | `planning/planners/mppi_planner.py` | MPPI trajectory optimization | calls `forward_prior` on a world model |
| `TwoLvlPlanner` | `planning/planners/two_lvl_planner.py:11` | Hierarchical MPPI wrapper | `l1_planner, l2_planner, l2_step_skip` |
| `Trainer` | `train.py:216` | Training loop, optimizer, checkpointing | `config: TrainConfig` |
| `Evaluator` | `evaluation/evaluator.py` | Dispatches planning + probing eval | `config: EvalConfig` |
| `DatasetFactory` | `data/dataset_factory.py:53` | Creates train/val/probe DataLoaders | `DataConfig` → `Datasets` namedtuple |

**Planning hierarchy** (`planning/planners/two_lvl_planner.py:25–85`):
1. L2 MPPI: optimizes 6 macro-actions (summed, 2D) over the L2 world model. First predicted L2 latent state becomes subgoal.
2. L1 MPPI: optimizes 15 primitive actions over the L1 world model targeting the L2 subgoal latent.
3. Subgoal transfer: `l2_result.pred_obs[1]` → passed as `goal` to L1 planner. Direct latent space matching (no decoder needed).

---

## 5. Config Surface

### Key L2 config knobs (`large_diverse_25maps_l2.yaml`)

| Parameter | Config key | Default (maze) | Effect |
|---|---|---|---|
| L2 step skip | `l2_step_skip` / `hjepa.step_skip` | 10 | Temporal downsampling factor |
| L2 trajectory length | `l2_n_steps` | 6 | Number of waypoints per L2 sample |
| Latent action dim | `hjepa.level2.predictor.z_dim` | 8 | Dimension of stochastic latent z |
| Action encoder arch | `hjepa.level2.predictor.action_encoder_arch` | `'8-64-8'` | MLP for posterior (not macro-action) |
| Posterior arch | `hjepa.level2.predictor.posterior_arch` | `'32-32'` | MLP mapping action → z distribution |
| Posterior input type | `hjepa.level2.predictor.posterior_input_type` | `'actions'` | What the posterior conditions on |
| Freeze L1 | `hjepa.freeze_l1` | `true` | Whether to freeze L1 during L2 training |
| L2 epochs | `epochs` | 5 | Training epochs for L2 |
| L2 batch size | `data.d4rl_config.batch_size` | 128 | |
| LR | `base_lr` | 0.01763 | Adam learning rate |
| Prediction loss coeff | `objectives_l2.prediction_obs.global_coeff` | 2.416 | |
| Proprio loss coeff | `objectives_l2.prediction_proprio.global_coeff` | 2.416 | |

### L1 config knobs (`large_diverse_25maps.yaml`)

| Parameter | Config key | Default | Effect |
|---|---|---|---|
| L1 trajectory length | `n_steps` | 15 | Timesteps per L1 sample |
| VICReg std coeff | `objectives_l1.vicreg_obs.std_coeff` | 29.4 | Variance regularization |
| VICReg cov coeff | `objectives_l1.vicreg_obs.cov_coeff` | 17.9 | Covariance regularization |
| IDM coeff | `objectives_l1.idm.coeff` | 4.81 | Inverse dynamics loss weight |
| Proprio coeff | `objectives_l1.prediction_proprio.global_coeff` | 2.42 | |

### Planning config knobs (inside `eval_cfg.h_d4rl_planning`)

| Parameter | Effect |
|---|---|
| `level1.mppi.num_samples` | L1 MPPI samples (easy: 500, hard: 1000) |
| `level2.mppi.num_samples` | L2 MPPI samples (easy/medium: 2000, hard: 4000) |
| `level2.mppi.noise_sigma` | L2 action noise std (10) |
| `replan_every` | Steps between replanning (4) |
| `max_plan_length_l2` | Max L2 plan length (35 for medium, 47 for hard) |

### Hardcoded values (not in config)
- Conv predictor layer specs: hardcoded dicts inside `conv_predictors.py:12–37` (e.g., `l2_d4rl_e_p: [(42, 32, 5, 1, 2), ...]`)
- L2 backbone: always `identity_encoder` (no additional encoding of L1 features)
- Summation action encoding: always sum, never learned
- `correct_l2_merge=False` default (renormalization path)

---

## 6. Paper-vs-Code Deltas

This repo implements **only the Diverse Maze / PLDM backend**. The paper's Franka experiments (VJEPA2-AC) and Push-T experiments (DINO-WM) use different codebases not present here. Some deltas below apply only to this backend.

### Delta 1: Action encoder — NOT a transformer with CLS token
- **Paper (§2.5):** "A learned action encoder that compresses sequences of primitive actions between waypoint states into latent macro-actions." Described as a transformer with CLS token (Appendix B.2: "Latent macro-actions are produced by a transformer-based action encoder A_ψ, where the CLS token is passed through an MLP head").
- **Code (maze backend):** `hjepa.py:101`: `l2_actions = actions.sum(dim=1)`. Fixed summation. No parameters. No transformer.
- **Location:** `pldm/models/hjepa.py:79–104`
- **What IS learned:** The posterior MLP (`posterior_arch: '32-32'`) maps the summed actions → stochastic latent z (8D). This z is the learned component, but it's a posterior model, not an encoder of the full action sequence.
- **Hypothesis:** The transformer action encoder was used for the Franka backend (VJEPA2-AC, not in this repo). The maze backend simplified this to summation.

### Delta 2: Loss is MSE (L2), not L1
- **Paper (Eq. 1):** `L_tf(ϕ,ψ) = (1/N) Σ ||ẑ_{t_{k+1}} - z_{t_{k+1}}||₁` — L1 norm.
- **Code:** `objectives/prediction.py:83`: `(encodings - predictions).pow(2).mean()` — MSE (L2 norm).
- **Applies to both L1 and L2 losses.**

### Delta 3: No explicit waypoint sampling
- **Paper:** "N=3 waypoint indices, middle waypoint chosen uniformly at random."
- **Code:** No random waypoint sampling. Instead, fixed temporal stride (`l2_step_skip=10`) produces equally-spaced waypoints. `l2_n_steps=6` waypoints per segment. No randomness in waypoint positions.
- **Location:** `pldm_envs/diverse_maze/d4rl.py:227–240`

### Delta 4: Latent action dimensionality
- **Paper:** "4D latent actions for Franka."
- **Code (maze):** `z_dim=8` in `large_diverse_25maps_l2.yaml`. Consistent with paper's Appendix B.4 ("8-dimensional latent action via an MLP").

### Delta 5: Stochastic latent z vs. deterministic latent action
- **Paper:** Describes deterministic latent macro-action `l_t = A_ψ(a_{t_k:t_{k+1}})`.
- **Code:** L2 predictor uses stochastic `z ~ posterior(actions)` with KL regularization. `z_stochastic=false` in config but `z_dim=8 > 0` still implies learned posterior sampling.
- UNCLEAR: with `z_stochastic=false`, it's unclear whether z is sampled (stochastic) or used as a deterministic posterior mean.

### Delta 6: High-level model architecture
- **Paper:** "Same ViT architecture as low-level model."
- **Code:** Same `conv2` architecture class (ConvPredictor), different subclass config (`l2_d4rl_e_p` vs `d4rl_b_p`). Architecture is not ViT at all — it's convolutional throughout. "Same ViT" claim applies to the Franka backend.

### Delta 7: VICReg in L2 training config
- `objectives_l2` in `large_diverse_25maps_l2.yaml` includes `vicreg_obs` with `std_coeff=35.026, cov_coeff=11.925` — but the `objectives` list only includes `PredictionObs` and `PredictionProprio`, not VICReg. So VICReg config is present but the objective is not in the active list.

### Undocumented tricks
- `correct_l2_merge` flag: if True, unnormalizes L1 actions before summing. Default False. Not mentioned in paper.
- L2 latent bounds: `normalizer.compute_l2_latent_bounds()` scans the full dataset to compute L2 latent stats for planning bounds. Computationally expensive; not mentioned in paper.
- `l2_use_latent_mean_std=false`: another planning bound option.
- `posterior_drop_p=0`: dropout on posterior — present as a config option but not used.

---

## 7. How to Run

**Prerequisites:**
1. Set `root_path` in YAML (currently `/scratch/wz1232/HWM_PLDM`) to your own path.
2. Generate or download maze dataset: see `pldm_envs/diverse_maze/data_generation/`.
3. Download pretrained checkpoints: `python pldm/download_ckpt_from_hf.py --out-dir ./pldm/pretrained`.

```bash
# Full training pipeline
# Step 1: Train L1 PLDM
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps.yaml \
  --values root_path=/your/path

# Step 2: Train L2 HWM (loading L1 checkpoint)
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values root_path=/your/path \
  load_checkpoint_path=/your/path/checkpoint/maze2d_large_diverse/<l1_ckpt>.ckpt

# Step 3: Evaluate L2 + L1 hierarchical planning
python pldm/train.py --configs pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values root_path=/your/path eval_only=true \
  load_checkpoint_path=/your/path/checkpoint/maze2d_large_diverse/l2_wo_encoder/<l2_ckpt>.ckpt
```

**Hardware:** GPU required. No explicit multi-GPU support noted in configs. W&B logging by default (`wandb=true`).

---

## 8. Friction Notes

### Tightly coupled / hard to modify

**Root path hardcoding:** `root_path: /scratch/wz1232/HWM_PLDM` appears in every config YAML and propagates to all dataset paths. Must be overridden at every invocation.

**L1 ↔ L2 dimension coupling:** `HJEPA.__init__` (line 50–73) infers L2 input dimensions from `level1.backbone.output_obs_dim`. If you change the L1 backbone, L2 must be reconfigured to match. There is no validation that they agree.

**Dataset ↔ normalizer ↔ model:** The `Normalizer` is built from dataset statistics, then passed into `HJEPA` which uses it inside `encode_actions`. This three-way dependency makes it hard to use the model without the full dataset pipeline.

**Config ↔ code consistency:** Config values like `n_steps % l2_step_skip == 0` are assumed but not validated. Silent failures if violated.

**L2 latent bounds:** `compute_l2_latent_bounds()` in `normalizer.py:448–560` requires a full pass over the dataset to compute stats for MPPI planning bounds. This creates a coupling between training and planning setup.

### Cleanly decoupled / easy to modify

**Objectives:** `objectives/` classes are independent callables on `ForwardResult`. New objectives can be added without touching the model code. Active objectives are listed in YAML under `objectives_l2.objectives`.

**Planners:** `planning/planners/` uses only `forward_prior()` as its model interface. You can swap MPPI for CEM or any other optimizer without retraining.

**Backbone ↔ predictor:** The backbone and predictor are independently configured via `JEPAConfig.backbone` and `JEPAConfig.predictor`. Different architectures compose freely.

### Dead code / unused branches
- Transformer predictor: `models/predictors/` has transformer variants but they are not used in any maze config.
- RNN predictor: configured but `rnn_layers=1` with no hidden state transfer between batches — effectively stateless.
- Ensemble predictor: `ensemble_size=1` by default; `vmap` code exists but inactive.
- Discrete latent z: `z_discrete=False` — discrete action space code present but unused.
- `forward_prior` on `HJEPA` is not implemented (raises `NotImplementedError` if `disable_l2=False`); planning calls `level1.forward_prior` and `level2.forward_prior` directly.
- `objectives_l2.vicreg_obs` in config has values set but `ObjectiveType.VICRegObs` is not in the active `objectives` list.
