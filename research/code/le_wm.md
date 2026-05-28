# LeWM (le-wm) Codebase Reference

**Scope:** Complete LeWorldModel codebase. Implements the full training and planning pipeline described in the paper. Flat, minimal structure — ~1400 lines of Python (excluding configs).

---

## 1. Layout

```
le-wm/
├── train.py          Training loop — Hydra entrypoint, data loading, loss, Lightning trainer (146 lines)
├── eval.py           Eval/planning loop — env simulation, CEM solver, metric logging (174 lines)
├── jepa.py           Core JEPA model: encode, predict, rollout, get_cost (154 lines)
├── module.py         Transformer primitives + SIGReg + AdaLN-zero components (286 lines)
├── utils.py          Image preprocessor, z-score normalizer, checkpoint callback (60 lines)
├── config/
│   ├── train/
│   │   ├── lewm.yaml            Primary training config (hyperparameters, optimizer, loss)
│   │   ├── model/lewm.yaml      Model architecture config (encoder, predictor, projectors)
│   │   ├── data/
│   │   │   ├── pusht.yaml       PushT dataset config
│   │   │   ├── dmc.yaml         DMControl Reacher config
│   │   │   ├── tworoom.yaml     TwoRoom navigation config
│   │   │   └── ogb.yaml         OGBench-Cube config
│   │   └── launcher/local.yaml  Local (non-SLURM) trainer config
│   └── eval/
│       ├── pusht.yaml, cube.yaml, tworoom.yaml, reacher.yaml   Per-env eval configs
│       ├── solver/cem.yaml      CEM planner config
│       └── launcher/local.yaml
├── assets/lewm.gif
└── README.md
```

---

## 2. Entry Points

```bash
# Train (env=pusht, dmc, tworoom, ogb)
python train.py data=pusht

# With overrides
python train.py data=pusht trainer.max_epochs=50 optimizer.lr=1e-4 loss.sigreg.weight=0.1

# Eval (config-name selects environment)
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# Random policy baseline
python eval.py --config-name=pusht.yaml policy=random
```

**Dataset location:** `$STABLEWM_HOME` (default `~/.stable-wm/`). Download HDF5 files from HuggingFace Hub as described in README.

**Config interaction:**
- Hydra entry `@hydra.main(config_path="./config/train", config_name="lewm")` in `train.py:47`.
- Config merge order: `lewm.yaml` (base) ← `launcher/local.yaml` ← `data/pusht.yaml` ← `model/lewm.yaml`.
- Dynamic resolution: `${embed_dim}` and `${eval:'${num_preds} + ${history_size}'}` evaluated at config compose time.
- Action encoder `input_dim` is set dynamically at runtime (`train.py:68`): `frameskip × dataset.get_dim("action")`.

---

## 3. Forward + Backward Trace

### 3.1 Data Loading (`train.py:53–79`)

```python
dataset = swm.data.load_dataset(dataset_name, ...)
# Returns HDF5 dataset; batches are dicts with keys: pixels, action, proprio, state
# Batch shape: {"pixels": (B, T, C, H, W), "action": (B, T, A*frameskip)}
# T = num_preds + history_size = 1 + 3 = 4 frames per sample
# action is frameskip=5 primitive actions concatenated per step → A_eff = A * 5
```

Transforms applied before batching:
1. `get_img_preprocessor()` → resize to 224×224, ImageNet normalization
2. `get_column_normalizer()` → z-score normalize actions, states

### 3.2 Full Forward Pass (teacher-forcing, one batch)

Entry: `lejepa_forward(self, batch, stage, cfg)` at `train.py:17–45`.

```
batch["pixels"]: (B, T=4, C=3, H=224, W=224)
batch["action"]: (B, T=4, A_eff)    # A_eff = frameskip × action_dim

Step 1: NaN cleaning (train.py:25)
  batch["action"] = torch.nan_to_num(batch["action"], 0.0)
  # Dataset artifact at trajectory boundaries; not mentioned in paper

Step 2: Encode  (train.py:27 → jepa.py:29–45)
  pixels_flat = rearrange(pixels, "b t c h w → (b t) c h w")  # (B*T, C, H, W)
  vit_out = encoder(pixels_flat)                                # ViT-Tiny forward
  cls_tok = vit_out.last_hidden_state[:, 0]                    # CLS token, (B*T, 192)
  emb_flat = projector(cls_tok)                                # MLP+BN1d, (B*T, 192)
  emb = rearrange(emb_flat, "(b t) d → b t d", b=B)           # (B, T, 192)
  act_emb = action_encoder(action)                              # Embedder: Conv1d+MLP, (B, T, 192)

Step 3: Slice context and target (train.py:32–35)
  ctx_emb = emb[:, :3]      # (B, 3, 192)  — history_size=3 frames as context
  ctx_act = act_emb[:, :3]  # (B, 3, 192)
  tgt_emb = emb[:, 1:]      # (B, 3, 192)  — shifted by num_preds=1

Step 4: Predict  (train.py:36 → jepa.py:47–55)
  preds = predictor(ctx_emb, ctx_act)     # ARPredictor + AdaLN: (B, 3, 192)
  pred_emb = pred_proj(preds)             # MLP+BN1d: (B, 3, 192)

Step 5: Losses  (train.py:39–41)
  pred_loss   = (pred_emb - tgt_emb).pow(2).mean()          # MSE, scalar
  sigreg_loss = sigreg(emb.transpose(0, 1))                  # SIGReg on (T, B, D)
  total_loss  = pred_loss + 0.09 * sigreg_loss               # λ=0.09 default

Backward: total_loss.backward()
  Updates: encoder θ, projector, predictor ϕ, pred_proj, action_encoder
  No frozen parameters, no stop-gradient, no EMA
```

### 3.3 SIGReg Implementation (`module.py:10–36`)

```python
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        t = torch.linspace(0, 3, knots)          # quadrature nodes (NOTE: [0,3], not [0.2,4])
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt                     # trapezoid rule endpoint correction
        window = torch.exp(-t.square() / 2.0)    # Gaussian kernel w(t)=exp(-t²/2)
        self.register_buffer("t", t)
        self.register_buffer("weights", weights * window)   # combined w·window

    def forward(self, proj):
        # proj: (T, B, D)  — all timesteps and batch samples jointly
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))           # random unit-norm directions, (D, M)
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (T, B, M, knots) — projected embeddings × nodes
        err = (x_t.cos().mean(-3) - self.phi).square()  + x_t.sin().mean(-3).square()
        # err: empirical char fn error vs. standard Gaussian char fn e^{-t²/2}
        statistic = (err @ self.weights) * proj.size(-2)   # sum over knots, scale by B
        return statistic.mean()                             # average over projections M and batch B
```

**Key detail:** SIGReg operates on `emb.transpose(0,1)` = `(T, B, D)` — it sees **all timesteps jointly**, not per-timestep independently. The aggregate distribution across the full sequence is pushed toward N(0,I).

---

## 4. Module Map

### jepa.py — `JEPA` class (`jepa.py:11–154`)

| Method | Lines | Role | Shapes |
|---|---|---|---|
| `encode(info)` | 29–45 | pixels → CLS → projector → emb; action → act_emb | pixels (B,T,C,H,W) → emb (B,T,D) |
| `predict(emb, act_emb)` | 47–55 | context emb + actions → predicted next emb | (B,T,D),(B,T,D) → (B,T,D) |
| `rollout(info, actions, history_size)` | 61–110 | multi-step autoregressive rollout for planning | actions (B,S,H,A) → preds (B,S,H,D) |
| `criterion(info)` | 112–126 | MSE cost between last predicted emb and goal emb | (B,S) scalar cost per candidate |
| `get_cost(info, actions)` | 128–153 | load to device, encode goal, rollout, return cost | actions (B,S,H,A) → cost (B,S) |

### module.py — Primitives

| Class/Function | Lines | Role | Key interface |
|---|---|---|---|
| `SIGReg` | 10–36 | Gaussian regularizer (see §3.3) | `(T,B,D)` → scalar |
| `FeedForward` | 38–53 | Transformer MLP (LN→Linear→GELU→Linear→Dropout) | `(B,T,D)` → `(B,T,D)` |
| `Attention` | 56–85 | Scaled dot-product attention, `is_causal=True` | `(B,T,D)` → `(B,T,D)` |
| `ConditionalBlock` | 88–111 | AdaLN-zero transformer block with action conditioning | `x(B,T,D), c(B,T,D)` → `(B,T,D)` |
| `Block` | 114–128 | Standard (no-conditioning) transformer block | `(B,T,D)` → `(B,T,D)` |
| `Transformer` | 131–187 | Stack of Block or ConditionalBlock | `x(B,T,in), c(B,T,in)` → `(B,T,out)` |
| `Embedder` | 189–214 | Action encoder: Conv1d projection + MLP | `(B,T,A)` → `(B,T,emb_dim)` |
| `MLP` | 217–241 | Projector: Linear → BN1d → GELU → Linear | `(B*T,D)` → `(B*T,D)` |
| `ARPredictor` | 244–285 | ViT-S predictor with AdaLN-zero action conditioning | `ctx_emb(B,T,D), ctx_act(B,T,D)` → `(B,T,D)` |
| `modulate(x, shift, scale)` | 6–8 | AdaLN modulation: `x*(1+scale)+shift` | element-wise |

### `ConditionalBlock` detail (`module.py:88–111`)

The action conditioning works via AdaLN-zero:
```python
# adaLN_modulation: SiLU → Linear(dim → 6*dim), zero-initialized
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
    self.adaLN_modulation(c).chunk(6, dim=-1)
x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
```

Zero-init (`module.py:101–103`):
```python
nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
```
At init, all gates = 0, so action conditioning has zero effect. Conditioning grows during training.

### `Embedder` — Action Encoder (`module.py:189–214`)

```python
# Architecture: Conv1d → MLP
# Input: (B, T, A_eff)  where A_eff = frameskip × action_dim
# Conv1d: (B, A_eff, T) → (B, smoothed_dim, T) → (B, T, smoothed_dim)
# MLP: (B, T, smoothed_dim) → (B, T, emb_dim)
```

This is the action encoder — a lightweight Conv1d + MLP, not a transformer. `input_dim` is set at runtime to `frameskip × action_dim`.

### `MLP` — Projector (`module.py:217–241`)

```python
# Architecture: Linear(in→hid) → BatchNorm1d(hid) → GELU → Linear(hid→out)
# Input: (B*T, D)  — must be flattened before calling
# Used as both 'projector' (post-encoder) and 'pred_proj' (post-predictor)
```

BatchNorm1d is set via config (`model/lewm.yaml:35`), not hardcoded in class definition (default in `MLP.__init__` is LayerNorm).

---

## 5. Config Surface

### Effective hyperparameters

| Parameter | Config key | Default | Sensitivity |
|---|---|---|---|
| SIGReg weight λ | `loss.sigreg.weight` | **0.09** | Only effective hyperparameter; robust in [0.01, 0.2] |
| Embedding dim d | `embed_dim` | 192 | Saturates above ~192; below ~96 hurts |
| Predictor depth | `model.predictor.depth` | 6 | ViT-S size; larger doesn't help |
| Predictor heads | `model.predictor.heads` | 16 | — |
| Predictor dropout | `model.predictor.dropout` | 0.1 | Critical: without it (p=0) PushT drops from 96→78% |
| History length N | `history_size` | 3 | Context frames passed to predictor |
| Learning rate | `optimizer.lr` | 5e-5 | AdamW |
| Batch size | `loader.batch_size` | 128 | — |
| Frame skip | `data.dataset.frameskip` | 5 | Effective time resolution |
| SIGReg projections M | `loss.sigreg.kwargs.num_proj` | 1024 | Insensitive; no tuning needed |
| SIGReg knots | `loss.sigreg.kwargs.knots` | 17 | Insensitive; no tuning needed |

### Planning config

| Parameter | Config key | Default | Location |
|---|---|---|---|
| Planning horizon H | `plan_config.horizon` | 5 | eval/pusht.yaml |
| Action block (frameskip) | `plan_config.action_block` | 5 | eval/pusht.yaml |
| Receding horizon | `plan_config.receding_horizon` | 5 | eval/pusht.yaml — executes ALL H steps before replanning |
| CEM samples | `solver.num_samples` | 300 | eval/solver/cem.yaml |
| CEM iterations | `solver.n_steps` | 30 | eval/solver/cem.yaml |
| CEM elite count | `solver.topk` | 30 | eval/solver/cem.yaml |

### What is hardcoded (not in config)

| Item | Location | Value |
|---|---|---|
| SIGReg quadrature range | `module.py:16` | `linspace(0, 3, knots)` — paper says [0.2, 4] |
| Gaussian kernel | `module.py:20` | `exp(-t²/2)` — weighting function w(t) |
| AdaLN output channels | `module.py:99` | 6×dim (fixed 3 params per sub-block × 2 sub-blocks) |
| Projector activation | `module.py:237` | GELU |
| AdaLN activation | `module.py:98` | SiLU |
| Attention causality | `module.py:83` | `is_causal=True` |
| NaN replacement value | `train.py:25` | 0.0 |
| Positional embedding init | `module.py:266` | `randn(1, num_frames, input_dim)` |

---

## 6. Paper-vs-Code Deltas

### Delta 1: Pseudocode bug — prediction loss
- **Paper Alg. 1**: `pred_loss = F.mse_loss(emb[:, 1:] − next_emb[:, :-1])` — subtracts two tensor slices, then passes result to `F.mse_loss`. This is not valid PyTorch (mse_loss needs two args, not one).
- **Actual code (`train.py:39`):** `(pred_emb - tgt_emb).pow(2).mean()` — equivalent to MSE between two tensors.
- **Impact:** The math is correct; the pseudocode is just poorly typeset. Not a real divergence.

### Delta 2: SIGReg quadrature range
- **Paper (Appendix A):** "T nodes uniformly distributed in [0.2, 4]."
- **Code (`module.py:16`):** `torch.linspace(0, 3, knots)` — range is [0, 3], not [0.2, 4].
- **Impact:** Minor quantitative difference in which frequencies the test statistic covers. Ablations show the method is insensitive to knot count, suggesting range also has low impact.

### Delta 3: SIGReg applied globally, not per-timestep
- **Paper (Section 3.1):** "latent embeddings are projected onto M random directions... aggregating these statistics."
- **Code (`train.py:40`):** `sigreg(emb.transpose(0, 1))` — inputs shape `(T, B, D)`. SIGReg averages `.mean(-3)` over T and B jointly. This is a global statistic over all timesteps and batch elements.
- **Impact:** Temporal structure is not individually regularized. The Gaussian prior applies to the aggregate distribution across all timestep positions, not to each timestep separately. This is why temporal straightening can emerge — temporal ordering is unconstrained.

### Delta 4: MPC replanning strategy
- **Paper:** "receding-horizon MPC" — implies step-by-step replanning.
- **Code (`eval/pusht.yaml`):** `receding_horizon=5` — executes the entire 5-step plan before replanning (not single-step receding horizon).
- **Impact:** Errors compound over 5 steps without correction. This is a weaker MPC regime than step-by-step.

### Delta 5: NaN action handling
- **Paper:** Not mentioned.
- **Code (`train.py:25`):** `batch["action"] = torch.nan_to_num(batch["action"], 0.0)` — replaces NaNs at trajectory boundaries with zero actions.
- **Impact:** Replaces missing actions with "do nothing" actions at trajectory boundaries. A reasonable heuristic but not discussed.

### Delta 6: AdaLN activation function
- **Paper:** Does not specify the activation function used in the AdaLN modulation network.
- **Code (`module.py:98`):** Uses SiLU (not GELU). The `FeedForward` blocks in the transformer use GELU. AdaLN modulation specifically uses SiLU (following DiT paper convention).

### Undocumented tricks
- **Projector symmetry:** Both `projector` (post-encoder) and `pred_proj` (post-predictor) have the same architecture and BatchNorm1d. This symmetric structure is not mentioned but may help because SIGReg is applied to encoder outputs — the projector maps to a space the regularizer can effectively act on.
- **BatchNorm1d via config override:** The `MLP` class defaults to `LayerNorm`, but the config overrides to `BatchNorm1d`. This override is a load-bearing config choice that affects whether SIGReg gradients flow correctly.
- **Gradient clipping:** `gradient_clip_val=1.0` in `lewm.yaml:22`. Not mentioned in the paper.
- **AdaLN conditions on action emb, not raw actions:** The predictor receives processed `act_emb` (from `Embedder`), not raw action tensors.

---

## 7. How to Run

**Setup:**
```bash
pip install stable-worldmodel stable-pretraining  # external dependencies
export STABLEWM_HOME=/path/to/datasets

# Download dataset (example: PushT)
# Follow README.md instructions to download HDF5 files
```

**Training:**
```bash
python train.py data=pusht                          # PushT (2D manipulation)
python train.py data=tworoom                        # TwoRoom navigation
python train.py data=ogb                            # OGBench-Cube (3D)
python train.py data=dmc                            # DMControl Reacher

# Override key hyperparameters
python train.py data=pusht loss.sigreg.weight=0.05 embed_dim=384 trainer.max_epochs=200
```

**Checkpoints:** saved to `$STABLEWM_HOME/checkpoints/{run_name}/` via `SaveCkptCallback`.

**Evaluation:**
```bash
# PushT
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# OGBench-Cube
python eval.py --config-name=cube.yaml policy=ogbench/cube_single_expert/lewm

# Reacher
python eval.py --config-name=reacher.yaml policy=dmc/reacher_random/lewm

# TwoRoom
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
```

Policy path `pusht/lewm` resolves to `$STABLEWM_HOME/pusht/lewm_object.ckpt`. Pretrained checkpoints from the paper are available via the repo's HuggingFace Hub (see README.md).

---

## 8. Friction Notes

### Tightly coupled / hard to modify

**`lejepa_forward` — loss is hardcoded** (`train.py:17–45`): The entire forward pass (encode → predict → MSE + SIGReg) is baked into one function. Changing the loss requires rewriting this function. There is no plugin mechanism for loss terms.

**BatchNorm1d dependency of SIGReg:** The projector MLP must use BatchNorm1d (not LayerNorm) for SIGReg gradients to flow correctly (paper's explicit requirement). This is set via config, but the coupling is invisible in the code — if you change `norm_fn` to LayerNorm in config, training will silently degrade.

**AdaLN hardwired to 6 channels** (`module.py:99`): `Linear(dim, 6*dim)`. Two sub-blocks (attn + mlp), three parameters each (shift/scale/gate). If you add more sub-blocks or change conditioning, you must rewrite `ConditionalBlock`.

**Action encoder `input_dim` runtime coupling** (`train.py:68`): `cfg.model.action_encoder.input_dim = frameskip × action_dim`. This is set at runtime by mutating the Hydra config. If you run eval with different frameskip or action_dim than training, the instantiated model will mismatch the checkpoint.

**SIGReg quadrature fixed** (`module.py:15–22`): The knot placement, weights, and kernel are all computed at `__init__` and registered as buffers. Changing the quadrature scheme requires subclassing.

### Cleanly decoupled / easy to modify

**Encoder is fully swappable** via `model/lewm.yaml encoder._target_`. ViT-Tiny is specified by string; any HuggingFace ViT or torchvision backbone with a `last_hidden_state[:, 0]` CLS output would work by changing the config.

**CEM solver is external** (`stable_worldmodel.solver.CEMSolver`). The JEPA only exposes `get_cost(info, actions) → cost`. You can swap in any optimizer that accepts a cost function without touching JEPA code.

**Dataset loading is abstracted** via `stable_worldmodel.data.load_dataset()`. New environments need a dataset YAML and HDF5 file; no code changes required.

**All hyperparameters config-driven.** λ, embed_dim, depth, history_size, frameskip, dropout are all overridable from CLI without code changes.

### Natural seams (without judgment)

1. **enc_θ ↔ projector:** CLS token → MLP+BN → z_t. The projector is symmetrically paired with pred_proj; both are needed for SIGReg to work.
2. **z_t ↔ predictor:** Clean — predictor consumes z_t as embedding sequence. The shared projector architecture between encoder and predictor output creates a symmetry.
3. **predictor ↔ CEM:** Clean — planning calls `get_cost()` which internally uses `rollout()`. Planning is fully decoupled from training.
4. **SIGReg ↔ enc_θ output:** SIGReg is called directly on encoder output embeddings. The BatchNorm1d in the projector is a prerequisite for this to work correctly.
5. **`lejepa_forward` ↔ Lightning:** Tight — this function is passed as a callback to `spt.Module`. Its signature must match the `spt.Module` convention.

### Dead code / unused branches

- `Transformer.__init__` supports both `Block` and `ConditionalBlock` via `block_class` parameter, but in practice only `ConditionalBlock` is used in `ARPredictor`. `Block` (standard transformer block) exists but is not used in any eval config.
- `jepa.py:detach_clone()` (`lines 8–9`): utility used in `rollout()` for inference, not in training.
- `eval.py` imports `gradient` solver as possible alternative to CEM, but only CEM configs are provided.
