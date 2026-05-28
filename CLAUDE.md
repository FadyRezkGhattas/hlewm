# Research Context: HWM & LeWM

This directory contains two papers and their codebases for world-model-based planning via JEPA-style latent prediction. Read this file first; then follow pointers below as needed.

---

## Papers

| Paper | PDF | Summary |
|---|---|---|
| **HWM** — Hierarchical Planning with Latent World Models | `hwm.pdf` | Adds a high-level world model (macro-actions over waypoints) on top of any existing latent WM; enables hierarchical CEM/MPPI planning; solves non-greedy tasks where flat planners fail. |
| **LeWM** — LeWorldModel | `le-wm.pdf` | End-to-end JEPA trained from raw pixels; collapse prevented by SIGReg (Gaussian regularizer); no EMA, no frozen encoder; 15M params, trains on one GPU; 48× faster planning than DINO-WM. |

Full paper notes: [`research/papers/hwm.md`](research/papers/hwm.md) | [`research/papers/lewm.md`](research/papers/lewm.md)

---

## Codebases

| Repo | Paper | Language | Key modules |
|---|---|---|---|
| `HWM_PLDM/` | HWM (Diverse Maze backend) | Python | `pldm/models/hjepa.py` — 2-level wrapper; `pldm/models/jepa.py` — single-level JEPA; `pldm/planning/planners/two_lvl_planner.py` — hierarchical MPPI; `pldm/objectives/prediction.py` — MSE loss |
| `le-wm/` | LeWM | Python | `jepa.py` — encode/predict/rollout/cost; `module.py` — SIGReg + AdaLN-zero transformer; `train.py` — training loop; `eval.py` — CEM planning |

Full code notes: [`research/code/hwm_pldm.md`](research/code/hwm_pldm.md) | [`research/code/le_wm.md`](research/code/le_wm.md)

### HWM_PLDM — Key module roles

| File:Line | Module | Role |
|---|---|---|
| `pldm/models/hjepa.py:28` | `HJEPA` | Two-level container: `level1` (frozen PLDM) + `level2` (trainable high-level WM) |
| `pldm/models/hjepa.py:79` | `HJEPA.encode_actions` | **Sums** primitive action chunks → L2 macro-actions (not a learned transformer; maze-specific) |
| `pldm/models/hjepa.py:122` | `HJEPA.forward_posterior` | Training forward: optionally runs L1, always runs L2 on L1-encoded waypoints |
| `pldm/models/jepa.py:39` | `JEPA` | Single-level world model: backbone encoder + predictor; used for both L1 and L2 |
| `pldm/planning/planners/two_lvl_planner.py:11` | `TwoLvlPlanner` | Hierarchical MPPI: L2 plan → subgoal → L1 MPPI to reach subgoal |
| `pldm/objectives/prediction.py:39` | `PredictionObjective` | MSE loss: `(encodings - predictions).pow(2).mean()` |
| `pldm/objectives/vicreg.py` | `VICRegObjective` | VICReg anti-collapse for L1 training |
| `pldm_envs/diverse_maze/d4rl.py:17` | `D4RLDataset` | Loads offline maze trajectories; produces L1 + L2 (chunked) samples |
| `pldm_envs/utils/normalizer.py:48` | `Normalizer` | Normalizes states, actions, L2 latents; passed into HJEPA |
| `pldm/train.py:216` | `Trainer` | Training loop, optimizer, checkpointing; config via YAML + OmegaConf |

### le-wm — Key module roles

| File:Line | Module | Role |
|---|---|---|
| `jepa.py:11` | `JEPA` | Top-level model: wraps encoder, predictor, projectors, action encoder |
| `jepa.py:29` | `JEPA.encode` | pixels → ViT-Tiny → CLS token → MLP+BN → z_t; also encodes actions via Embedder |
| `jepa.py:47` | `JEPA.predict` | context emb + act emb → ARPredictor → pred_proj → ẑ_{t+1} |
| `jepa.py:61` | `JEPA.rollout` | Multi-step autoregressive prediction for CEM planning |
| `jepa.py:128` | `JEPA.get_cost` | Encode goal, rollout, return MSE cost — interface to CEM solver |
| `module.py:10` | `SIGReg` | Gaussian regularizer: projects embeddings onto random directions, minimizes Epps-Pulley normality statistic |
| `module.py:88` | `ConditionalBlock` | AdaLN-zero transformer block: action embedding generates shift/scale/gate via zero-init linear |
| `module.py:189` | `Embedder` | Action encoder: Conv1d + MLP maps raw actions → action embeddings |
| `module.py:217` | `MLP` | Projector (with BatchNorm1d): used post-encoder and post-predictor |
| `module.py:244` | `ARPredictor` | ViT-S autoregressive predictor with causal masking + AdaLN-zero conditioning |
| `train.py:17` | `lejepa_forward` | Loss function: encode → predict → MSE + λ·SIGReg; called by Lightning |

---

## Research Docs

| File | Contents |
|---|---|
| [`research/comparison.md`](research/comparison.md) | HWM vs LeWM side-by-side on all key design axes; prose on conceptual differences |
| [`research/glossary.md`](research/glossary.md) | Every overloaded term (latent, target, context, predictor, world model, level, plan, macro-action…) defined per-paper |
| [`research/open_questions.md`](research/open_questions.md) | What neither paper resolves; direction-agnostic |
| [`research/notes_corrections.md`](research/notes_corrections.md) | Corrections to `jepa_notes.md` with paper/code citations |

Direction-specific work lives under [`research/directions/`](research/directions/). That folder may opine freely; everything else in `research/` stays neutral.

---

## Critical Paper-vs-Code Deltas (read before editing code)

**HWM_PLDM (maze backend):**
- Action encoder = fixed sum (`hjepa.py:101`), not transformer. Transformer described in paper applies to Franka backend (different repo).
- Loss = MSE (`prediction.py:83`), not L1 as paper states.
- No explicit waypoint sampling; uses fixed temporal stride `step_skip=10`.
- `root_path` hardcoded in every YAML as `/scratch/wz1232/HWM_PLDM` — must override.

**le-wm:**
- SIGReg runs on all timesteps jointly (`emb.transpose(0,1)` = `(T,B,D)`), not per-timestep.
- SIGReg quadrature range is `[0,3]` in code (`module.py:16`), not `[0.2,4]` as paper states.
- MPC executes full 5-step plan before replanning (not step-by-step receding horizon).
- BatchNorm1d in projector is set by config (`model/lewm.yaml:35`), not code default; changing it silently breaks SIGReg.
- AdaLN uses SiLU (not GELU).

---

## Quick orientation

**If you're exploring HWM:** Start at `research/papers/hwm.md` for the paper, then `research/code/hwm_pldm.md` for the maze codebase. Note this repo covers only the maze (PLDM) backend; Franka and Push-T use different codebases.

**If you're exploring LeWM:** `research/papers/lewm.md` then `research/code/le_wm.md`. The entire training loop is in `train.py:lejepa_forward` (~25 lines). Core architecture is in `module.py`.

**If you want to compare them:** `research/comparison.md`. **If a term is ambiguous:** `research/glossary.md`.

# Behavioural Guidelines for Implementation

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

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

## 4. Goal-Driven Execution

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
