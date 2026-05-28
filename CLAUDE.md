# le-wm Codebase Guide

This directory implements **LeWM** (flat, single-level) and **HLEWM** (hierarchical two-level extension).
Read this file before editing any code. It is self-contained — you do not need the parent `research/` directory.

---

## What this codebase does

**LeWM (flat):** End-to-end JEPA world model trained from raw pixels. Encoder + predictor trained jointly; collapse prevented by SIGReg (Gaussian regularizer). Plans via flat CEM in 192-dim latent space. 15M params, single GPU, few hours.

**HLEWM (hierarchical):** Adds a second-level world model on top of frozen L1 LeWM. L2 plans at a longer timescale via macro-action embeddings → produces subgoals in the shared latent space → L1 CEM reaches each subgoal with primitive actions. Motivation: flat CEM accumulates prediction errors over long horizons and fails on non-greedy tasks. Hierarchy reduces the number of sequential prediction steps needed to reach distant goals.

---

## File map

| File | Role |
|---|---|
| `jepa.py` | `JEPA`: encode/predict/rollout/get_cost. Flat L1 world model. |
| `module.py` | `SIGReg`, `ARPredictor`, `Embedder`, `MLP`, `MacroActionEncoder`, `Transformer`, `ConditionalBlock` |
| `train.py` | `lejepa_forward` (L1), `l2_forward` (L2), `_build_hjepa`. Switches on `train_level` config flag. |
| `hjepa.py` | `HJEPA`: two-level wrapper. Frozen L1 + trainable L2 predictor + MacroActionEncoder. |
| `eval.py` | Flat L1 CEM evaluation via `WorldModelPolicy`. Unchanged from original LeWM. |
| `heval.py` | Hierarchical CEM evaluation. `SubgoalAdapter`, `HierarchicalWorldModelPolicy`. |
| `utils.py` | `get_img_preprocessor`, `get_column_normalizer`, `SaveCkptCallback` |
| `config/train/lewm.yaml` | L1 training config |
| `config/train/hlewm.yaml` | L2 training config (`train_level: 2`) |
| `config/train/data/pusht_l2.yaml` | PushT data config with `num_steps: 64` for L2 trajectory length |
| `config/eval/pusht.yaml` | Flat L1 eval config |
| `config/eval/hpusht.yaml` | Hierarchical eval config |

---

## Key module roles

### Flat LeWM (L1)

| File:Line | Module | Role |
|---|---|---|
| `jepa.py:11` | `JEPA` | Top-level model: encoder, predictor, projectors, action encoder |
| `jepa.py:29` | `JEPA.encode` | pixels → ViT-Tiny → CLS token → MLP+BN → z_t; also encodes actions via Embedder |
| `jepa.py:47` | `JEPA.predict` | context emb + act emb → ARPredictor → pred_proj → ẑ_{t+1} |
| `jepa.py:61` | `JEPA.rollout` | Multi-step autoregressive rollout for CEM |
| `jepa.py:112` | `JEPA.criterion` | MSE(pred_emb[-1], goal_emb[-1]) — terminal cost |
| `jepa.py:128` | `JEPA.get_cost_from_emb` | Like get_cost but takes precomputed goal embedding; used by L1 CEM at hierarchical inference |
| `jepa.py:147` | `JEPA.get_cost` | Encode goal pixels, rollout, return MSE cost — CEM interface |
| `module.py:10` | `SIGReg` | Gaussian regularizer for L1 training only. Projects embs onto 1024 random dirs, minimizes Epps-Pulley statistic |
| `module.py:88` | `ConditionalBlock` | AdaLN-zero transformer block. Action emb → shift/scale/gate (zero-init → neutral at start) |
| `module.py:189` | `Embedder` | L1 action encoder: Conv1d + MLP maps per-timestep actions → embeddings |
| `module.py:217` | `MLP` | Projector with BatchNorm1d: used post-encoder and post-predictor |
| `module.py:295` | `ARPredictor` | ViT-S causal predictor with AdaLN-zero conditioning |
| `train.py:17` | `lejepa_forward` | L1 loss: MSE + λ·SIGReg |

### HLEWM (L2 extension)

| File:Line | Module | Role |
|---|---|---|
| `module.py:217` | `MacroActionEncoder` | Transformer + CLS over variable-length primitive-action chunks → fixed macro embedding. **Training only** — not called at inference. |
| `hjepa.py:17` | `HJEPA` | Two-level container: frozen L1 JEPA + trainable L2 predictor + MacroActionEncoder |
| `hjepa.py:41` | `HJEPA.encode_waypoints` | Encodes waypoint pixels via frozen L1 encoder. Called at training AND inference. |
| `hjepa.py:58` | `HJEPA.encode_macro_actions` | **Training only.** Maps raw action chunks via MacroActionEncoder → macro embeddings. |
| `hjepa.py:73` | `HJEPA.predict` | Teacher-forced L2 prediction. Input: (waypoint_embs, macro_embs). MSE target. |
| `hjepa.py:97` | `HJEPA.rollout_l2` | **Inference only.** Takes macro embeddings directly (no encoding). CEM candidates ARE the macro embeddings. |
| `hjepa.py:130` | `HJEPA.get_cost` | Routes to get_l2_cost. Makes HJEPA compatible with WorldModelPolicy as L2 planner. |
| `hjepa.py:143` | `HJEPA.get_l2_cost` | L2 CEM cost: encode current obs + goal → rollout_l2 → terminal MSE |
| `train.py:55` | `l2_forward` | L2 loss: random waypoint sampling → MSE only, no SIGReg |
| `train.py` | `_build_hjepa` | Loads L1 from checkpoint, instantiates L2 components, returns HJEPA |
| `train.py` | `_build_hjepa_config` | Constructs full HJEPA OmegaConf config (with `_target_: hjepa.HJEPA`) for `save_pretrained`; overrides `l1_jepa._target_` to `jepa.JEPA` so `get_cost_from_emb` is available at eval. |
| `heval.py` | `SubgoalAdapter` | Wraps L1 JEPA. Intercepts get_cost → get_cost_from_emb(subgoal_emb). Uses a counter to index the correct per-env subgoal as the CEM iterates envs one at a time (batch_size=1). |
| `heval.py` | `HierarchicalWorldModelPolicy` | Extends BasePolicy. set_env configures both solvers. get_action: L2 CEM → subgoal → L1 CEM. |

---

## Critical design decisions (read before modifying anything)

### 1. MacroActionEncoder is training-only — CEM searches in macro embedding space directly

At **training**: `encode_macro_actions(action_chunks)` maps raw primitive-action chunks → macro embeddings via the transformer. L2 predictor learns to condition on these embeddings.

At **inference** (L2 CEM): `rollout_l2(wp0, macro_emb_sequence)` takes macro embeddings **directly** as CEM candidates. The CEM optimises in this `embed_dim=192` space without going through MacroActionEncoder. This is correct: the CEM searches in the same space that MacroActionEncoder maps to at training time.

**If you add a MacroActionEncoder call inside `rollout_l2`, you will break L2 CEM.** The L2 CEMSolver configures with `action_dim = 192` (macro embedding dim) and samples in that space.

### 2. Shared latent space — enforced by reusing frozen L1 encoder

L2 predictions must live in the same latent space as L1 embeddings so that L2 subgoals can be passed directly to L1 CEM as targets. This is guaranteed by having `encode_waypoints` use `self.l1_jepa.encoder` + `self.l1_jepa.projector`. **Do not introduce a separate L2 encoder** — it would produce embeddings in a different space, breaking subgoal transfer.

### 3. BatchNorm must stay in eval mode during L2 training

L1's projector has `BatchNorm1d`. During L2 training, the L1 encoder is frozen. Lightning calls `model.train()` at each epoch start, which would put BN into training mode and start updating its running statistics from L2 training data. This corrupts the BN statistics that L1 learned, silently shifting the latent space.

`HJEPA.train()` overrides this by always calling `self.l1_jepa.eval()` after `super().train(mode)`. **Do not remove this override.**

### 4. No SIGReg for L2 training

L2 training loss is MSE only (`l2_forward`). The L1 encoder is frozen and already produces roughly Gaussian-distributed embeddings (from SIGReg during L1 training). Adding SIGReg to L2 would be redundant and could distort the already-regularized geometry. The `spt.Module` for L2 training is constructed without a `sigreg=` argument.

### 5. Random waypoint sampling — why it requires the transformer action encoder

Waypoints are sampled randomly within a trajectory (fixed start and end, random intermediates). This means the primitive-action chunks between consecutive waypoints have **variable length** across different batches. The transformer+CLS in `MacroActionEncoder` handles variable lengths naturally. An MLP could not — it requires fixed input size.

Fixed-stride sampling (like HWM's maze backend) would allow an MLP. Random sampling is used here to expose the L2 model to diverse transition lengths, producing a more general high-level model.

### 6. Teacher-forcing only for L2 (no rollout loss)

`l2_forward` uses teacher-forcing: context = ground-truth waypoints, target = next ground-truth waypoint. There is no rollout loss where L2 predicts from its own outputs. This matches HWM's finding that L2 rollout loss does not improve performance and complicates training. The L2 predictor has never seen its own predictions during training; this is a known limitation.

### 7. L2 CEM fake action space

The `CEMSolver` gets its `action_dim` from `env.action_space.shape[1:]`. For L2, there is no real environment action space — macro actions are 192-dim embeddings. `HierarchicalWorldModelPolicy.set_env` configures the L2 solver with a fake `gymnasium.spaces.Box(shape=(1, 192))`, giving `_action_dim = 192` and with `action_block=1` → `action_dim = 192`. **If you change `embed_dim`, update `hpusht.yaml:macro_action_dim` too.**

### 8. Known failure mode: subgoal reachability

L2 CEM may plan latent subgoals that are geometrically valid in the Gaussian latent space but dynamically unreachable by L1 CEM. There is no formal reachability guarantee. If L1 consistently fails to reach L2 subgoals, the root cause is usually insufficient data coverage or a mismatch between L2's planned timescale and L1's planning horizon.

---

## Critical paper-vs-code deltas

**Flat LeWM:**
- SIGReg runs on all timesteps jointly (`emb.transpose(0,1)` = `(T,B,D)`), not per-timestep.
- SIGReg quadrature range is `[0,3]` in code (`module.py:16`), not `[0.2,4]` as paper states.
- MPC executes full 5-step plan before replanning (receding_horizon=5), not step-by-step.
- BatchNorm1d in projector is set by config (`model/lewm.yaml`); changing to LayerNorm silently breaks SIGReg gradient flow.
- AdaLN uses SiLU, not GELU.

**HLEWM (our extension, not in any paper):**
- Macro-action encoder is training-only. Papers (HWM Franka) describe a transformer encoder; our inference bypasses it entirely — CEM searches in macro embedding space directly.
- L2 training loss = MSE only. HWM paper describes L1 loss for Franka; our code uses MSE throughout (consistent with LeWM).
- Waypoint sampling = random intermediates in a range (matching HWM Franka). HWM maze code uses fixed stride=10 — that is maze-specific and not used here.
- No rollout loss for L2. HWM maze code uses rollout-only for both levels; our L2 uses teacher-forcing only.
- L2 CEM fake action space = `Box(shape=(1, 192))` — not described anywhere; derived from CEMSolver internals.

---

## Workflow

```bash
conda activate hlewm

# Train flat L1
python train.py data=pusht

# Evaluate flat L1 (use pre-trained HF checkpoint or local folder name after training)
python eval.py --config-name pusht policy=FadyRezk/lewm-pusht-fixed   # pre-trained
python eval.py --config-name pusht policy=lewm                         # after training

# Train L2 (l1_checkpoint = HF repo ID or local folder name under $STABLEWM_HOME/checkpoints/)
python train.py --config-name hlewm data=pusht_l2 l2.l1_checkpoint=FadyRezk/lewm-pusht-fixed

# Evaluate hierarchical (policy = folder name written by SaveCkptCallback after training)
python heval.py --config-name hpusht policy=hlewm

# Bootstrap HJEPA for pipeline testing (random L2 weights, no training needed)
python bootstrap_hjepa.py
python heval.py --config-name hpusht policy=hlewm
```

Checkpoint format (stable-worldmodel 0.1): `weights.pt` + `config.json` under
`$STABLEWM_HOME/checkpoints/<name>/`. Pass `<name>` or an HF repo ID as `policy=`.

Data sequence lengths:
- L1: `num_steps = history_size + num_preds = 4` (set in `data/pusht.yaml`)
- L2: `num_steps = 30` (= `num_waypoints × min_waypoint_gap`; PushT episodes cap at 246 raw frames → 49 frameskipped steps, so 30 is both the minimum and near the practical ceiling)

---

## stable-worldmodel interface notes

`CEMSolver.solve(info_dict)` returns `{'actions': tensor(B, H, action_dim), ...}`.

`WorldModelPolicy.get_action(info_dict)`:
- Manages per-env action buffer (deque)
- Replans when buffer is empty
- Calls `solver(sliced_info)` → buffers `receding_horizon` steps → pops one per step

`CEMSolver` calls `model.get_cost(expanded_info, candidates)` where:
- `expanded_info`: each tensor has shape `(B, S, ...)` (B envs, S samples)
- `candidates`: `(B, S, H, action_dim)`
- Returns: `(B, S)` cost

`PlanConfig` fields: `horizon`, `receding_horizon`, `action_block`, `history_len`, `warm_start`.

`CEMSolver.configure(action_space, n_envs, config)`:
- `_action_dim = np.prod(action_space.shape[1:])`
- `action_dim = _action_dim * config.action_block`

---

## Behavioural Guidelines for Implementation

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
