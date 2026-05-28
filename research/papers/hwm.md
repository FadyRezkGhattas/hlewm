# HWM: Hierarchical Planning with Latent World Models

**Citation:** Zhang et al., 2026. FAIR at Meta / NYU / Mila.
**Code:** `HWM_PLDM/` (this repo)
**Date:** 2026-04-06

---

## Core Claim

HWM frames hierarchical planning as a plug-in inference-time abstraction: train a second (high-level) latent world model that operates on compressed macro-actions and variable-stride waypoints, couple its predictions as subgoals into any existing low-level latent world model, and CEM plan over both levels — enabling zero-shot non-greedy long-horizon control where flat planners fail.

---

## Problem Setup

**Setting:** Offline, reward-free, goal-conditioned. MDP M = (S, A, µ, p). Agent has access to offline dataset D of trajectories τ = (s₁, a₁, s₂, ..., s_T). At eval time: given current observation s₁ and goal observation s_g, find actions to reach s_g.

**Inputs / Outputs:**

| Quantity | Shape / Type | Description |
|---|---|---|
| s_t = (x_t, p_t) | (256×256×3, R^7) | RGB image + 7D end-effector state |
| a_t | R^7 | Delta end-effector pose: a_t = p_{t+1} - p_t |
| z_t = E(s_t) | R^{256×1408} (Franka) | Frozen ViT-g/16 spatial feature map |
| l_t | R^4 (Franka) | Latent macro-action from action encoder |
| ẑ_{t+1} | same as z_t | Predicted next-latent from world model |

**What is optimized:** Two separate training objectives — one for the low-level world model P^(1) and one for the high-level world model P^(2) + action encoder A_ψ. No reward. Planning is a separate CEM optimization at inference time.

**Tensor shapes (Franka):**

- Encoder input: `[B, T, 256, 256, 3]` → `[B, T, 256, 1408]` (256 spatial tokens per frame)
- Low-level world model input: interleaved `{(z_k, p_k, a_k)}_{k∈[T]}`, context T=16
- High-level world model input: interleaved `{(l_{t_k}, z_{t_k})}_{k∈[N]}`, context T=3 waypoints
- Latent macro-action: `[4]` (scalar vector, not spatial)

---

## Architecture

```
Offline data
    │
    ▼
E (encoder, frozen V-JEPA 2 ViT-g/16)
    │  z_t ∈ R^{256×1408}
    ├──────────────────────────────────────────────────────────┐
    ▼                                                          ▼
P^(1): Low-Level World Model                    A_ψ: Action Encoder
~300M ViT (same arch as VJEPA2-AC predictor)   Transformer + CLS → MLP
Input: {z_k, p_k, a_k}_{k∈[T]}                Input: chunk of primitive actions
Output: ẑ_{k+1} (spatial, same shape as z_k)  Output: l_t ∈ R^4
    │                                                          │
    │                                                          ▼
    │                                          P^(2): High-Level World Model
    │                                          Same ViT architecture as P^(1)
    │                                          Input: {l_{t_k}, z_{t_k}}_{k∈[N]}
    │                                          Output: ẑ_{t_{k+1}} (predicted waypoints)
    │                                                          │
    └─────────────────────────────────────────────────────────┘
                         │
                   Hierarchical CEM planner
                   High: optimize l̂_{1:H}, get subgoals z̃_i
                   Low: optimize â_{1:h} to reach z̃_1
```

**Encoder:** Frozen V-JEPA 2 ViT-g/16 backbone (not trained in HWM). Maps each frame independently. Spatial feature map output (not CLS).

**Low-level world model P^(1):** Architecture identical to the underlying backbone's predictor (VJEPA2-AC, DINO-WM, or PLDM respectively). Block-causal attention: within a timestep all tokens attend to each other; across timesteps, causal. Actions, states, and feature maps are mapped to the predictor's hidden dim via learned affine transforms.

**Action encoder A_ψ:** Transformer that processes a variable-length sequence of primitive actions (a_{t_k}, ..., a_{t_{k+1}-1}) and outputs the CLS token, passed through an MLP head, to produce l_{t_k} ∈ R^4. Handles variable-length chunks, always outputs one fixed-size vector. Trained jointly with P^(2).

**High-level world model P^(2):** Same ViT architecture as the low-level model, but conditioned on latent macro-actions l instead of primitive actions. Takes interleaved (l_{t_k}, z_{t_k}) pairs as input. Predicts next waypoint latent ẑ_{t_{k+1}}. Operates in the same shared latent space as P^(1) — this is the architectural commitment that makes direct subgoal transfer possible.

---

## Losses

### Low-Level World Model (Appendix B.1)

**Teacher-forcing loss:**
```
ẑ_{k+1} := P^(1)_θ({a_t, z_t}_{t≤k})
L_tf(θ) = (1/T) Σ_{k=1}^{T} || ẑ_{k+1} - z_{k+1} ||_1
```

**Multi-step rollout loss (open-loop):**
```
L_roll(θ) = Σ_{j=2}^{T} || P^(1)_θ(a_{1:j}, z_1) - z_{j+1} ||_1
```
where P^(1)_θ(a_{1:j}, z_1) feeds the model's own predictions back as input.

**Total:**
```
L(θ) = γ_tf · L_tf(θ) + γ_roll · L_roll(θ)
```
For Franka: γ_tf = γ_roll = 1.0.

**Collapse prevention:** None beyond the frozen encoder. The frozen V-JEPA 2 backbone already provides non-collapsed representations; P^(1) only needs to learn transition dynamics on top.

### High-Level World Model (Eq. 1)

**Teacher-forcing only (no rollout loss):**
```
ẑ_{t_{k+1}} := P^(2)_ϕ({l_{t_i}, z_{t_i}}_{i≤k})
L_tf(ϕ, ψ) = (1/N) Σ_{k=1}^{N} || ẑ_{t_{k+1}} - z_{t_{k+1}} ||_1
```
L1 distance against ground-truth encoded waypoints. A_ψ and P^(2) are trained jointly.

**No rollout loss for high-level model** (γ_roll = 0.0 throughout; Franka, Push-T). Only 3 waypoints per segment (context T=3).

---

## Training Recipe

| Component | Franka | Push-T | Diverse Maze |
|---|---|---|---|
| Encoder | Frozen ViT-g/16 (V-JEPA 2) | Frozen DINOv2 ViT-S/14 | Trained from scratch (conv) |
| Low-level epochs | 200 | 100 | 3 |
| Low-level batch | 256 | 256 | 128 |
| High-level epochs | 120 | 500 | 5 |
| High-level batch | 768 | 128 | 128 |
| Context T (low) | 16 | 4 (stride 5) | 15 |
| Context T (high) | 3 waypoints | 5 waypoints | 6 waypoints |
| Latent action dim | 4 | — | 8 (MLP encoder, not transformer) |
| Waypoint stride | variable (up to 4s) | 25–70 steps | fixed stride 10 |
| γ_tf / γ_roll (low) | 1.0 / 1.0 | 1.0 / 0.0 | 0.0 / 1.0 |
| γ_tf / γ_roll (high) | 1.0 / 0.0 | 1.0 / 0.0 | 0.0 / 1.0 |

**Data (Franka):** ~96h DROID + 30h RoboSet, unlabeled real-robot manipulation. Spatial resolution 256×256. FPS uniform ∈ (3,10) for DROID, (1,5) for RoboSet. Random resize-and-crop augmentation, aspect ratio ∈ (0.75, 1.33).

**EMA / stop-grad:** None in HWM itself. V-JEPA 2 backbone was trained with EMA target encoder (standard JEPA), but that encoder is fully frozen when HWM trains. No EMA or stop-grad in P^(1), A_ψ, or P^(2).

**Waypoint sampling:** N=3 total waypoints. t₁ = trajectory start, t_N = trajectory end, middle waypoint uniform random. Variable-length segments (0.33–4 seconds for Franka). For Diverse Maze: fixed stride of 10 steps between 6 waypoints.

**Diverse Maze exception:** Encoder and low-level predictor are trained jointly from scratch using PLDM recipe (VICReg-based anti-collapse). High-level model reuses frozen low-level encoder (MLP action encoder, not transformer). Low-level loss is rollout-only (γ_tf=0, γ_roll=1).

---

## Eval Protocol + Results

**Planning:** CEM (Franka, Push-T) or MPPI (Diverse Maze). Replanning every k=1 step (MPC). Parallel GPU rollouts.

| Task | Flat WM | HWM | Δ |
|---|---|---|---|
| Franka pick-and-place (cup) | 0% | 70% | +70 |
| Franka pick-and-place (box) | 0% | 60% | +60 |
| Franka drawer | 30% | 70% | +40 |
| Push-T (d=25) | 84% | 89% | +5 |
| Push-T (d=50) | 55% | 78% | +23 |
| Push-T (d=75) | 17% | 61% | +44 |
| Diverse Maze (D∈[5,8]) | 100% | 100% | 0 |
| Diverse Maze (D∈[9,12]) | 63% | 95% | +32 |
| Diverse Maze (D∈[13,16]) | 44% | 83% | +39 |

Planning compute: HWM matches or exceeds flat planner success rates with ~3–4× less wall-clock planning time.

Latent action dimensionality ablation: 4D optimal for Franka. Below 4D, high-level planner fails to produce valid plans. Above 4D, subgoals become unreachable by the low-level planner.

High-level prediction accuracy vs low-level: high-level single-step predictions are more accurate than low-level multi-step rollouts for horizons ≥ 1.5s.

---

## Abstraction Structure

**What exists:**

| Component | Responsibility | Coupling |
|---|---|---|
| E (encoder) | Maps obs → latent z | Shared across both WMs; frozen in Franka/Push-T, trained with P^(1) in Maze |
| P^(1) (low-level WM) | Short-horizon latent dynamics, conditioned on primitive actions | Depends on E; reuses backbone's architecture and training recipe exactly |
| A_ψ (action encoder) | Compresses variable-length primitive action sequence → fixed macro-action | Trained jointly with P^(2); architecture is a transformer (Franka/Push-T) or MLP (Maze) |
| P^(2) (high-level WM) | Long-horizon latent dynamics on waypoints, conditioned on macro-actions | Same architecture as P^(1); shares E with P^(1) |
| CEM/MPPI planner | Inference-time trajectory optimization | Stateless; wraps both WMs; hyperparameters swept per task and horizon |
| ViT decoder | Decode z → RGB for visualization only | Not used in planning or training |

**What the paper presents as modular:** The framework is explicitly a plug-in on top of any existing latent world model. P^(1) is always taken from an existing baseline unchanged. Only A_ψ and P^(2) are new components.

**What is entangled:**
- The high-level and low-level WMs must share the same encoder and latent space for subgoal transfer to work. You cannot swap one encoder without retraining both.
- The latent action dimensionality (4D) was hand-chosen by ablation for Franka; there is no automatic way to determine it.
- Action encoder architecture differs per setting (transformer for Franka/Push-T, MLP for Maze) — the "architecture-agnostic" claim applies at the WM level, not the action encoder level.
- High-level planner horizon H and low-level planner horizon h are independent hyperparameters tuned per domain.

**Natural seams (without judgment):**
1. E ↔ P^(1): clean — E is frozen; P^(1) only needs z_t as input.
2. P^(1) ↔ CEM: clean — planner calls P^(1) as a black box rollout function.
3. A_ψ ↔ P^(2): fused — trained jointly; CLS token projection from A_ψ feeds directly into P^(2)'s input format.
4. P^(2) ↔ P^(1): coupled only through the shared latent space z; otherwise architecturally independent.
5. High-level CEM ↔ Low-level CEM: coupled only through subgoal handoff (z̃_1 from high passed as target to low).

---

## Limitations

**Authors acknowledge:**
- Performance degrades at very long horizons for all methods.
- Strictly top-down hierarchy: no feedback from low-level to high-level; failures don't trigger re-planning at the high level until the next replanning step.
- Uncertainty-aware planning is absent.
- Need for more abstract representations for very high-level reasoning.

**Additional observations:**
- High-level model trained with teacher forcing only (γ_roll=0): it has never seen its own predictions during training, potentially making open-loop rollout in CEM less accurate.
- Latent action dimensionality (4D) is a free design parameter with no principled derivation; the ablation shows it matters significantly.
- No formal criterion for subgoal reachability: the model may propose subgoals that are geometrically plausible in latent space but dynamically unreachable.
- CEM sample counts differ widely by domain (3000 high-level for Franka vs 900 for Push-T); sensitivity to these hyperparameters is not fully characterized.
- The approach is evaluated on three substantially different backbone world models, but they are never compared head-to-head in the same domain with the same data.
- Visualization decoder (ViT decoder for Franka) is trained separately on DROID; its quality depends on encoder reconstruction fidelity, which is not a training objective.

---

## Open Questions and Ambiguities

- **Waypoint selection strategy:** The middle waypoint is chosen uniformly at random. UNCLEAR: does the distribution of intermediate waypoints affect the kinds of behaviors the high-level model learns to represent? Alternative strategies (e.g., highest-variance states, equidistant in latent space) are not explored.
- **High-level rollout loss:** Why is γ_roll=0 for the high-level model? If low-level rollout loss is important (it is, for Franka), why not for the high-level? This is not discussed.
- **Action encoder capacity:** Transformer vs MLP depending on domain — is this a principled choice or convenience?
- **UNCLEAR:** The paper says the high-level planner "typically produces valid plans" when latent action dim ≥ 4, but what counts as "valid" is defined via cosine similarity to expert behavior, not via actual reachability. These two definitions may diverge.
- **Joint optimization:** The high and low levels are trained independently and planned top-down. The authors flag this as a limitation but do not propose or analyze alternatives.
- **UNCLEAR:** In the Diverse Maze setting, the high-level encoder is frozen from the low-level model. In the Franka setting, both share the same frozen V-JEPA 2 encoder. In Push-T, the low-level uses DINOv2. These are three different encoder-training regimes; the effect of this variation is not isolated.
