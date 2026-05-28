# LeWM: LeWorldModel — Stable End-to-End JEPA from Pixels

**Citation:** Maes*, LeLidec*, Scieur, LeCun, Balestriero. 2026. Mila / NYU / Brown.
**Code:** `le-wm/` (this repo)
**arXiv:** 2603.13912 (March 2026 preprint)

---

## Core Claim

LeWM is the first JEPA-style world model that trains end-to-end from raw pixels — no frozen encoder, no EMA, no stop-gradient — using exactly two loss terms: MSE next-embedding prediction and SIGReg (a statistical regularizer enforcing isotropic Gaussian latents). This reduces the number of tunable loss hyperparameters from six (PLDM) to one (λ), with a provable anti-collapse guarantee, while matching or beating prior methods at 48× faster planning speed.

---

## Problem Setup

**Setting:** Offline, reward-free, goal-conditioned control. Agent observes raw pixel sequences and associated actions from a fixed offline dataset; no environment interaction during training.

**Inputs / Outputs:**

| Quantity | Shape / Type | Description |
|---|---|---|
| o_t | R^{224×224×3} | Raw RGB frame |
| a_t | R^A | Continuous action (domain-specific) |
| z_t = enc_θ(o_t) | R^d, d=192 default | Latent embedding ([CLS] token + projection) |
| ẑ_{t+1} = pred_ϕ(z_t, a_t) | R^d | Predicted next-step latent |
| z_g = enc_θ(o_g) | R^d | Goal latent |

**What is optimized:** Encoder θ and predictor ϕ jointly, end-to-end, via two-term loss. No additional networks (no target encoder, no IDM, no decoder).

**Tensor shapes:**
- Encoder input: `[B, T, C, H, W]` = `[128, 4, 3, 224, 224]`
- Encoder output: `[B, T, D]` = `[128, 4, 192]` (CLS tokens after projection)
- Predictor input: `[B, T, D]` embeddings with actions fused via AdaLN at each layer
- Predictor output: `[B, T, D]` = next-step predictions (with causal masking)
- SIGReg input: `Z ∈ R^{N×B×d}` (history × batch × dim)

Frame skip of 5: each "action" is actually a block of 5 primitive environment actions. Effective prediction horizon = 5 env steps per predictor step.

---

## Architecture

```
o_t ──► Encoder (ViT-Tiny, ~5M)
        patch_size=14, 12 layers, 3 heads, dim=192
        [CLS] token → 1-layer MLP + BatchNorm → z_t ∈ R^192
              │
              ▼
        Predictor (ViT-S, ~10M)
        6 layers, 16 heads, 10% dropout
        Action conditioning: AdaLN at each layer (init to zero)
        Causal masking over history of N frames
        Projector head (same MLP+BN as encoder projector) → ẑ_{t+1}
              │
              ▼
        L_pred = MSE(ẑ_{t+1}, z_{t+1})       [detached from z_{t+1}? No — end-to-end]
        SIGReg(Z) across all (t, batch) embeddings
```

**Encoder detail:** ViT-Tiny (HuggingFace), patch size 14, 12 layers, 3 attention heads, hidden dim 192. The observation embedding z_t is the [CLS] token from the last layer, passed through a 1-layer MLP with BatchNorm. BatchNorm is required (not LayerNorm) because the final ViT layer applies LayerNorm, which prevents SIGReg from being optimized effectively — without BN after LN, gradients for SIGReg do not flow properly.

**Predictor detail:** ViT-S backbone with learned positional embeddings and causal masking over the observation history. History length N=3 for PushT and OGBench-Cube; N=1 for TwoRoom. Actions incorporated via Adaptive Layer Normalization (AdaLN) at each transformer layer — each layer's scale and shift are learned functions of a_t. AdaLN parameters initialized to zero so action conditioning starts neutral and grows during training. Followed by the same projector as the encoder.

**No target encoder.** Both z_{t+1} (ground truth) and ẑ_{t+1} (predicted) come from the same online encoder. The predictor is optimizing toward a moving target that it itself influences through backpropagation.

---

## Losses

### Training Objective (Eq. 3)

```
L_LeWM = L_pred + λ · SIGReg(Z)
```

**Prediction loss (teacher-forcing):**
```
L_pred = || ẑ_{t+1} - z_{t+1} ||²₂
       = MSE(pred_ϕ(z_t, a_t), enc_θ(o_{t+1}))
```

Gradients flow through both the predictor (via ẑ_{t+1}) and the encoder (via z_{t+1}). No stop-gradient on either side.

**SIGReg (Eq. 2, Appendix A):**

Sketched-Isotropic-Gaussian Regularizer. Forces the joint distribution of latent embeddings to match N(0, I).

Step 1 — Project: sample M=1024 random unit-norm directions u^(m) ∈ S^{d-1}. Project embeddings: h^(m) = Z u^(m), where Z ∈ R^{N×B×d}.

Step 2 — Test normality: apply Epps-Pulley univariate test statistic T(·) to each projection:
```
T(m) = ∫_{-∞}^{∞} w(t) |ϕ_N(t; h^(m)) - ϕ_0(t)|² dt
```
where ϕ_N is the empirical characteristic function of h^(m) and ϕ_0 is the standard Gaussian characteristic function. Integral computed via trapezoid quadrature over T nodes in [0.2, 4].

Step 3 — Aggregate:
```
SIGReg(Z) = (1/M) Σ_{m=1}^{M} T(m)
```

**Anti-collapse guarantee (Cramér-Wold theorem):** SIGReg(Z) → 0 ⟺ P_Z → N(0, I). Matching all 1D marginals implies matching the joint distribution. This is a provable convergence statement (in distribution), unlike EMA/stop-grad which lack formal collapse guarantees.

**Collapse prevention mechanism:** SIGReg is the sole collapse prevention. It is applied independently at each timestep (not across the temporal dimension), which leaves temporal structure unconstrained.

**Hyperparameters:**
- M (number of projections): insensitive, performance flat from 64–1024. Not an effective hyperparameter.
- T (integration knots): insensitive, performance flat. Not an effective hyperparameter.
- λ (regularization weight): the only effective hyperparameter. Performance robust in [0.01, 0.2]; degrades at λ=0.5 (regularizer dominates dynamics modeling). Can be found with bisection search (O(log n)) rather than grid search.

---

## Training Recipe

**Data:**
- TwoRoom: 10,000 episodes, avg 92 steps, heuristic policy. 10 epochs.
- PushT: 20,000 expert episodes, avg 196 steps. 10 epochs (sufficient; matches DINO-WM results).
- OGBench-Cube: 10,000 episodes, 200 steps, heuristic data-collection policy. 10 epochs.
- Reacher: 10,000 episodes, 200 steps, SAC policy. 10 epochs.

**Processing:** Frame skip 5 (actions grouped into blocks of 5). Batch size 128. Sub-trajectories of length 4 frames. Resolution 224×224.

**Optimizer:** Not explicitly named; `stable-pretraining` library.

**Hardware:** Single NVIDIA L40S GPU. Training completes in "a few hours."

**EMA / stop-grad:** None. Gradients propagate through all components end-to-end.

**AdaLN initialization:** Scale and shift parameters initialized to zero, ensuring action conditioning grows progressively from zero impact.

**Predictor dropout:** 10% dropout (p=0.1). Ablation: critical — without dropout (p=0), performance drops from 96% to 78% on PushT. Dropout rates above 0.1 also hurt.

**No reconstruction loss.** Ablation: adding a decoder reconstruction objective slightly hurts downstream planning (96% → 86% on PushT). The JEPA objective already captures the information needed; reconstruction encourages irrelevant visual details.

**Encoder architecture ablation:** ResNet-18 works nearly as well as ViT-Tiny (94% vs 96% PushT). Method is largely architecture-agnostic.

**Embedding dimensionality:** Performance saturates around d=192. Below ~96, performance degrades significantly.

---

## Eval Protocol + Planning

**Planning (inference):** CEM trajectory optimization in latent space.
- 300 candidate action sequences sampled per CEM iteration
- 30 iterations (PushT), 10 iterations (other envs)
- Top 30 elites update Gaussian parameters
- Planning horizon H=5 steps = 25 env steps (frame skip 5)
- MPC with full-horizon execution before replanning (execute all 5 steps, then replan)
- Goal cost: C(ẑ_H) = ||ẑ_H - z_g||²₂ (MSE in latent space, terminal only)

**Evaluation:** Trajectories sampled from offline dataset. Initial state random, goal = state 25–100 steps later. Binary success.

**Results:**

| Task | LeWM | PLDM | DINO-WM | DINO-WM+prop |
|---|---|---|---|---|
| TwoRoom | 20% | 87% | 79% | — |
| Reacher | 100% | 100% | 100% | — |
| PushT | 86% | 78% | — | 79% |
| OGBench-Cube | 74% | 65% | — | 92% |

**Planning speed:** Full plan ≈ 0.98s (vs DINO-WM ≈ 47s). ~48× faster. LeWM uses ~200× fewer tokens than DINO-WM (CLS only vs patch tokens). PLDM has similar speed to LeWM.

**Fixed-FLOP comparison:** At equal compute budget, LeWM significantly outperforms DINO-WM (13% vs 90% PushT; 48% vs 74% OGBench-Cube) — because DINO-WM's large encoder dominates compute.

**Training stability:** LeWM achieves higher PushT success with lower variance across seeds than PLDM (96%±2.83 vs 78%±5.0).

**Physical probing (PushT):** Linear probes recover agent location (r=0.974), block location (r=0.986), block angle (r=0.902). Competitive with DINO-WM; substantially better than PLDM on linear probes.

**Violation-of-expectation:** LeWM assigns significantly higher prediction error (surprise) to physically impossible events (object teleportation) vs. visual perturbations (color change) vs. unperturbed trajectories. Statistically significant (paired t-test p<0.01) for teleportation perturbations.

**Emergent temporal straightening:** Latent trajectories become increasingly linear over training (cosine similarity between consecutive velocity vectors approaches 1). Emerges without any explicit temporal regularization — surpasses PLDM which has an explicit L_time-sim term.

---

## Abstraction Structure

**What exists:**

| Component | Responsibility | Coupling |
|---|---|---|
| enc_θ | Maps o_t → z_t (CLS token) | Trained jointly with pred_ϕ; must use BN+MLP projector (not just LN) |
| pred_ϕ | Models z_t + a_t → ẑ_{t+1} | Action fused via AdaLN (not concatenation or separate embedding) |
| SIGReg | Collapse prevention | Applied to enc_θ output; random projections resampled each step |
| CEM planner | Inference-time trajectory optimizer | Calls pred_ϕ as black box; CEM is the only planner used |
| Decoder (viz) | o reconstruction from z (diagnostic only) | Not trained with model; attached post-hoc for visualization |

**What the paper presents as modular:**
- Encoder architecture (ViT or ResNet) is swappable.
- Action space is generic (AdaLN handles arbitrary action dims).
- SIGReg is an independent regularization module from a separate paper (Le-JEPA).

**What is entangled:**
- Encoder output must go through BatchNorm+MLP projector (not optional — the BN is required for SIGReg gradient flow, and LN alone breaks it).
- Both encoder and predictor share projector architecture; they are symmetrically coupled.
- SIGReg is computed on enc_θ outputs only, not on predictor outputs — if you change what the encoder outputs, SIGReg behavior changes.
- AdaLN initialization to zero is load-bearing: without it, training is unstable.
- The CLS-only representation (no spatial tokens) is a design choice that trades off spatial resolution for planning speed. This affects what information is available for planning.

**Natural seams (without judgment):**
1. enc_θ ↔ pred_ϕ: relatively clean — enc_θ produces z_t, pred_ϕ consumes z_t. Both share projector architecture.
2. SIGReg ↔ enc_θ: enc_θ output goes directly into SIGReg. The BN layer in the projector is tightly required by SIGReg.
3. pred_ϕ ↔ CEM: clean — CEM calls pred_ϕ as a black box rollout engine during planning.
4. enc_θ + pred_ϕ ↔ offline data: clean — no environment interaction, pure offline batch training.

---

## Limitations

**Authors acknowledge:**
- TwoRoom performance is poor (20%): SIGReg's isotropic Gaussian prior is hard to satisfy when data has low intrinsic dimensionality. Enforcing high-dimensional Gaussianity on an inherently low-dimensional task can produce poorly structured representations.
- Flat planning only: long-horizon reasoning requires hierarchical world modeling (flagged as future work).
- Relies on offline datasets with sufficient coverage; data diversity affects SIGReg effectiveness.
- Requires action labels; future work: inverse dynamics modeling to learn latent actions.
- No large-scale pretraining; still domain-specific training.

**Additional observations:**
- CLS-only encoder: no spatial tokens available at planning time. Cannot distinguish spatially fine-grained states that have similar global statistics.
- Teacher-forcing only during training (no rollout loss): the model is never exposed to its own prediction errors during training, yet is asked to use its outputs autoregressively at test time.
- AdaLN initialization to zero is a training stability hack — without this, training apparently fails; this fragility is not fully characterized.
- SIGReg is applied per-timestep independently, leaving temporal correlations unconstrained. Temporal straightening emerges, but there is no guarantee it results in a semantically useful temporal ordering.
- On OGBench-Cube, LeWM (74%) is notably below DINO-WM (92%), suggesting the CLS-only representation struggles with visually complex 3D environments. Block orientation / yaw is not well-recovered by any method.
- The MPC strategy here executes the full planned horizon (H=5) before replanning — unlike HWM which replans every step. This means errors accumulate over 5 steps without correction.
- UNCLEAR: the paper does not specify the optimizer, learning rate, or weight decay.

---

## Open Questions and Ambiguities

- **SIGReg vs. useful non-Gaussian structure:** The Gaussian prior is isotropic and unimodal. Real environment latent spaces may have multimodal or non-Gaussian geometry (e.g., two physically disconnected regions). Is SIGReg pushing toward a geometry that conflicts with environment structure in some cases? The Two-Room failure may be an instance of this.
- **UNCLEAR:** Exactly why is BatchNorm (rather than any other normalization) required before SIGReg? The paper says LN prevents SIGReg from being "optimized effectively" but does not give a precise mechanism.
- **Temporal SIGReg:** SIGReg is applied per-timestep. Could applying it across the temporal dimension change the latent geometry? The paper notes temporal straightening emerges without this — would explicit temporal SIGReg hurt or help?
- **Predictor size:** ViT-S is optimal; ViT-B hurts slightly. The explanation offered (optimization stability) is informal. UNCLEAR what the fundamental bottleneck is.
- **Two-Room failure:** If the environment has low intrinsic dimensionality, the Gaussian prior forces the encoder to use more dimensions than the environment warrants. The paper observes but does not resolve this. Does this generalize to any environment where d >> intrinsic_dim?
- **Scaling:** LeWM is 15M parameters. How does performance scale with model size? No scaling experiments are presented.
- **Real-robot applicability:** All environments are simulated. The gap between simulated data and real-robot deployment (noise, latency, distribution shift) is not addressed.
- **UNCLEAR:** The paper's pseudocode (Alg. 1) shows `pred_loss = F.mse_loss(emb[:, 1:] − next_emb[:, :-1])` — this subtracts rather than computes the mean, suggesting a typo or unusual convention. The mathematical formulation (Eq. 1) clearly states `||ẑ_{t+1} - z_{t+1}||²₂`.
