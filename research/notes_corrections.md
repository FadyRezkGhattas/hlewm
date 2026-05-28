# Notes Corrections

Corrections and addenda to `jepa_notes.md`. The notes are generally accurate on V-JEPA, V-JEPA 2, and V-JEPA-AC. Most issues are in the HWM section (§5). The LeWM paper is not covered by the notes.

Notation: **[WRONG]** = factually incorrect; **[IMPRECISE]** = correct in spirit but misleading in detail; **[MISSING]** = important detail omitted; **[OK]** = verified correct.

---

## §1 I-JEPA — Verified Correct

The description of the three-component architecture (context encoder, target encoder, predictor), EMA sensitivity, and tensor shapes is accurate. No corrections needed.

---

## §2 V-JEPA — Verified Correct

Spatial/temporal tokenization, the 3D convolution patch embedding, input/output shapes, and the masking strategy are all accurately described. No corrections needed.

---

## §3 V-JEPA 2 — Verified Correct

The description of scaling, 3D RoPE, and progressive resolution training is consistent with what the papers cite.

---

## §4 V-JEPA-AC — Minor Imprecision

**[IMPRECISE]** The notes describe the predictor as "~300M parameter transformer (24 layers, 16 heads, 1024 hidden dimension)." The HWM paper (Appendix B.2) refers to this as a "~300M-parameter ViT" following the VJEPA2-AC setup but does not restate the exact layer/head count in HWM itself. The numbers (24 layers, 1024 dim) are likely correct from the V-JEPA 2 paper, but they are not cited in HWM directly.

**[OK]** The block-causal attention description is accurate. Within-timestep: full attention. Across-timestep: causal.

**[OK]** The 7D end-effector state and delta-pose actions are confirmed in Appendix B.2: `p_k ∈ R^7`, `a_k = p_{k+1} - p_k`.

---

## §5 HWM — Main Corrections

### CEM sample counts

**[WRONG]** The notes describe CEM as: "Sample 200 candidate action sequences... Take top 20."

**Actual numbers (HWM paper, Table 9 and Table 10):**
- Single-level Franka planner: **2400 samples, 20 elites, 15 iterations**
- Hierarchical high-level planner: **3000 samples, 22 elites, 15 iterations**
- Hierarchical low-level planner: **800 samples, 12 elites, 5 iterations**

The "200 candidates / top 20" figures appear to be a simplified illustrative example in the notes, not the actual experimental values. The elite count of 20 is close to the actual 22 (high-level) and 20 (single-level), but the sample count of 200 is an order of magnitude off from 3000.

*Citation: HWM paper Tables 9–10, Appendix C.*

### High-level model training loss

**[MISSING]** The notes say the high-level model uses "Same L1 teacher-forcing loss against the ground-truth encoded waypoints." This is correct for the loss *type*, but omits a critical detail:

**The high-level model has no rollout loss (γ_roll = 0.0 throughout all domains).** The low-level model uses γ_roll = 1.0 (Franka) — a full multi-step rollout loss — because the teacher-forcing gap at inference time is severe for short-horizon models. The high-level model skips this, meaning it is only trained on teacher-forcing and has never seen its own predictions during training, despite being rolled out autoregressively during CEM planning at test time.

*Citation: HWM paper Appendix B.2, Table 6: `γ_roll = 0.0` for high-level across all settings.*

### Action encoder

**[IMPRECISE]** The notes say the action encoder is "a transformer with a CLS token" and outputs a "4D in the Franka experiments." Both are accurate per the paper. However:

1. The maze backend in `HWM_PLDM/` implements the action encoder as a **fixed sum** (`hjepa.py:101`), not a transformer. The transformer action encoder described in the paper applies to the Franka backend (V-JEPA 2-AC codebase, not in this repo).

2. The maze code uses `z_dim=8` (8D latent action) via a stochastic posterior MLP — a different architecture than either the 4D transformer encoder or the sum encoder described elsewhere.

*Citation: `HWM_PLDM/pldm/models/hjepa.py:101`; HWM paper Appendix B.4 ("8-dimensional latent action via an MLP").*

### Loss function norm

**[MISSING]** The notes do not specify whether the L1 or L2 norm is used. The HWM paper states L1:

> `L_tf(ϕ,ψ) = (1/N) Σ_{k=1}^{N} || ẑ_{t_{k+1}} - z_{t_{k+1}} ||₁` (Eq. 1)

However, the `HWM_PLDM` maze codebase uses **MSE (L2 norm)**: `(encodings - predictions).pow(2).mean()` (`objectives/prediction.py:83`). This is a paper-vs-code discrepancy in the maze backend. The Franka backend (V-JEPA 2-AC) likely uses L1 as stated in the paper.

### Number of waypoints in maze code

**[IMPRECISE]** The notes describe N=3 waypoints "with the middle waypoint chosen uniformly at random." This matches the Franka training description. But:

- **Franka (paper, Appendix B.2):** N=3, middle waypoint random. ✓
- **Push-T (paper, Appendix B.3):** N=5 waypoints.
- **Maze (paper, Appendix B.4):** N=6 waypoints with **fixed stride of 10 steps** (not random).
- **Maze (code, `large_diverse_25maps_l2.yaml`):** `l2_n_steps=6`, `l2_step_skip=10` — confirmed fixed stride, no random sampling.

The notes correctly describe the Franka setup but don't note that the maze uses a different (fixed-stride) scheme.

*Citation: HWM paper Appendix B.2–B.4; `HWM_PLDM/pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml`.*

---

## §5 HWM — Verified Correct

**[OK]** The hierarchical planning description (high-level CEM → subgoals → low-level CEM → execute first action → replan) is accurate.

**[OK]** The motivation for hierarchy ("pick-and-place goes from 0% to 70%") matches Table 1 in the paper.

**[OK]** The description that "high-level predictions are more accurate for horizons ≥ 1.5s" matches Figure 6 in the paper.

**[OK]** The latent action dimensionality ablation description (4D optimal, too low = invalid plans, too high = unreachable subgoals) matches Figure 7 in the paper.

**[OK]** The 4D latent action dimension for Franka is confirmed in both paper and planning config.

---

## LeWM — Not Covered in Notes

The notes do not discuss LeWM at all. Key facts for future reference:

- **Not a V-JEPA descendant in the usual sense:** LeWM does not use EMA, stop-gradient, or frozen encoders. It is an end-to-end JEPA trained entirely from scratch.
- **SIGReg:** The sole collapse prevention mechanism. Imported from companion paper "Le-JEPA" (Balestriero & LeCun 2025, arXiv:2511.08544).
- **Encoder output:** CLS token only (192D), not spatial feature map. Much more compact than V-JEPA 2.
- **Planning speed:** ~1s full plan vs. V-JEPA 2 / DINO-WM's ~47s. The speedup comes from fewer tokens, not from a better algorithm.
- **Comparison to PLDM:** LeWM is the clean successor to PLDM — same end-to-end philosophy, simpler training objective (2 terms vs. 7 terms).
