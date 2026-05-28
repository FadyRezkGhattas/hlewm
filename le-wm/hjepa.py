"""Hierarchical JEPA (HJEPA): two-level latent world model.

L1 (low-level):  existing flat JEPA, frozen after checkpoint load.
L2 (high-level): new ARPredictor trained on waypoint sequences;
                 macro-actions encoded by MacroActionEncoder.

Both levels share the same encoder and latent space, so L2 subgoals
are directly usable as L1 CEM targets without any projection.

Training vs. inference distinction for macro-actions:
  - Training: MacroActionEncoder maps raw primitive-action chunks → macro embeddings.
  - Inference: L2 CEM searches directly in macro embedding space (same dim as embed_dim).
    MacroActionEncoder is NOT called during rollout — candidates are already embeddings.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class HJEPA(nn.Module):

    def __init__(self, l1_jepa, macro_action_encoder, l2_predictor, l2_pred_proj):
        super().__init__()
        self.l1_jepa = l1_jepa
        self.macro_action_encoder = macro_action_encoder
        self.l2_predictor = l2_predictor
        self.l2_pred_proj = l2_pred_proj

        # Freeze L1 immediately at construction.
        self.l1_jepa.requires_grad_(False)

    def train(self, mode=True):
        # Lightning calls model.train() at each epoch start.
        # Always keep l1_jepa in eval mode so BN running stats don't drift.
        super().train(mode)
        self.l1_jepa.eval()
        self.l1_jepa.requires_grad_(False)
        return self

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_waypoints(self, pixels):
        """Encode waypoint observations using the frozen L1 encoder.

        pixels: (B, N, C, H, W)
        returns: (B, N, D)
        """
        B, N = pixels.shape[:2]
        flat = rearrange(pixels, "b n ... -> (b n) ...")
        flat = flat.float()
        output = self.l1_jepa.encoder(flat, interpolate_pos_encoding=True)
        emb = output.last_hidden_state[:, 0]          # CLS token: (B*N, D)
        emb = self.l1_jepa.projector(emb)             # (B*N, D)
        return rearrange(emb, "(b n) d -> b n d", b=B, n=N)

    def encode_macro_actions(self, action_chunks):
        """Encode a list of variable-length primitive-action chunks (training only).

        action_chunks: list of (N-1) tensors, each (B, L_k, action_dim)
        returns: (B, N-1, macro_dim)

        At inference, L2 CEM bypasses this — it searches directly in macro embedding
        space without going through the action encoder.
        """
        macro_embs = [self.macro_action_encoder(chunk) for chunk in action_chunks]
        return torch.stack(macro_embs, dim=1)          # (B, N-1, macro_dim)

    # ------------------------------------------------------------------
    # L2 prediction (training)
    # ------------------------------------------------------------------

    def predict(self, wp_embs, macro_embs):
        """Teacher-forced L2 prediction.

        wp_embs:    (B, N, D)    — encoded waypoints (all N, including target)
        macro_embs: (B, N-1, D)  — macro-action embeddings from MacroActionEncoder
        returns:    (B, N-1, D)  — predicted next-waypoint embeddings
        """
        ctx_embs = wp_embs[:, :-1]                     # (B, N-1, D)
        preds = self.l2_predictor(ctx_embs, macro_embs)  # (B, N-1, D)
        preds = self.l2_pred_proj(
            rearrange(preds, "b t d -> (b t) d")
        )
        return rearrange(preds, "(b t) d -> b t d", b=wp_embs.size(0))

    # ------------------------------------------------------------------
    # L2 rollout (inference / CEM)
    # ------------------------------------------------------------------

    def rollout_l2(self, wp_emb_0, macro_emb_sequence, history_size=3):
        """Autoregressive L2 rollout for CEM.

        At inference, the L2 CEM directly optimises in macro embedding space.
        Candidates are already macro embeddings — MacroActionEncoder is NOT called here.

        wp_emb_0:           (B, 1, D)         — initial encoded waypoint
        macro_emb_sequence: (B, S, H, D)      — CEM candidates (macro embeddings)
        history_size: int                      — L2 predictor context length

        returns: (B, S, H, D) — predicted waypoint embeddings
        """
        B, S, H, _ = macro_emb_sequence.shape

        emb = wp_emb_0.unsqueeze(1).expand(B, S, -1, -1)   # (B, S, 1, D)
        emb = rearrange(emb, "b s t d -> (b s) t d").clone()
        macros = rearrange(macro_emb_sequence, "b s h d -> (b s) h d")  # (BS, H, D)

        preds = []
        HS = history_size
        for t in range(H):
            macro_emb = macros[:, t:t + 1]              # (BS, 1, D)
            ctx = emb[:, -HS:]                           # (BS, HS, D)
            # Expand macro to match context length for AdaLN conditioning
            ctx_act = macro_emb.expand(-1, ctx.size(1), -1)  # (BS, HS, D)

            pred = self.l2_predictor(ctx, ctx_act)[:, -1:]  # (BS, 1, D)
            pred = self.l2_pred_proj(
                rearrange(pred, "b t d -> (b t) d")
            ).unsqueeze(1)                               # (BS, 1, D)
            preds.append(pred)
            emb = torch.cat([emb, pred], dim=1)

        pred_stack = torch.cat(preds, dim=1)             # (BS, H, D)
        return rearrange(pred_stack, "(b s) h d -> b s h d", b=B, s=S)

    # ------------------------------------------------------------------
    # CEM cost functions (inference)
    # ------------------------------------------------------------------

    def get_cost(self, info_dict, macro_action_candidates):
        """L2 CEM cost: optimise macro-action embedding sequences to reach goal.

        Named get_cost so WorldModelPolicy can use HJEPA as a drop-in model.
        Routes to get_l2_cost.

        info_dict must contain:
            "pixels":  (B, 1, T, C, H, W)  current observation frames
            "goal":    (B, 1, T, C, H, W)  goal observation frames
        macro_action_candidates: (B, S, H, macro_dim) — CEM candidates in macro emb space
        """
        return self.get_l2_cost(info_dict, macro_action_candidates)

    def get_l2_cost(self, info_dict, macro_action_candidates):
        """L2 CEM cost function.

        Returns (B, S) terminal MSE cost.
        Stores predicted waypoint embeddings in info_dict["l2_pred_embs"].

        macro_action_candidates: (B, S, H, macro_dim) — already in macro embedding space
        """
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        macro_action_candidates = macro_action_candidates.to(device)

        # Encode current observation (most recent frame as L2 starting waypoint)
        cur_pixels = info_dict["pixels"][:, 0, -1:]       # (B, 1, C, H, W)
        wp_emb_0 = self.encode_waypoints(cur_pixels)       # (B, 1, D)

        # Encode goal
        goal_pixels = info_dict["goal"][:, 0, -1:]        # (B, 1, C, H, W)
        goal_emb = self.encode_waypoints(goal_pixels)      # (B, 1, D)

        # L2 rollout: candidates are already macro embeddings
        pred_embs = self.rollout_l2(wp_emb_0, macro_action_candidates)  # (B, S, H, D)
        info_dict["l2_pred_embs"] = pred_embs

        # Terminal cost: MSE between last predicted waypoint and goal
        B, S = pred_embs.shape[:2]
        goal_expanded = goal_emb.unsqueeze(1).expand(B, S, -1, -1)  # (B, S, 1, D)
        cost = F.mse_loss(
            pred_embs[:, :, -1, :],
            goal_expanded[:, :, 0, :].detach(),
            reduction="none",
        ).sum(dim=-1)  # (B, S)

        return cost

    # ------------------------------------------------------------------
    # Training forward (bound to spt.Module at training time)
    # ------------------------------------------------------------------

    def training_forward(self, batch, stage):
        """L2 training: random waypoint sampling → MSE only, no SIGReg.

        Called as an unbound method: spt.Module(forward=HJEPA.training_forward).
        spt.Module binds itself as `self`, so self here is the Lightning module.
        Access the HJEPA model via self.model; config via self.cfg.
        """
        import random
        cfg = self.cfg
        T = batch["pixels"].size(1)
        N = cfg.l2.num_waypoints
        min_gap = cfg.l2.min_waypoint_gap

        pool = list(range(min_gap, T - min_gap))
        intermediates = sorted(random.sample(pool, N - 2))
        wp_indices = [0] + intermediates + [T - 1]

        pixels_wp = batch["pixels"][:, wp_indices]

        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        action_chunks = [
            batch["action"][:, wp_indices[k]:wp_indices[k + 1]]
            for k in range(N - 1)
        ]

        wp_embs = self.model.encode_waypoints(pixels_wp)
        macro_embs = self.model.encode_macro_actions(action_chunks)
        pred_embs = self.model.predict(wp_embs, macro_embs)
        tgt_embs = wp_embs[:, 1:].detach()

        loss = (pred_embs - tgt_embs).pow(2).mean()
        self.log(f"{stage}/l2_pred_loss", loss.detach(), on_step=True, sync_dist=True)
        return {"loss": loss}

    def get_l1_cost(self, info_dict, action_candidates, subgoal_emb):
        """L1 CEM cost function: reach a subgoal embedding with primitive actions.

        Bypasses goal-pixel encoding — subgoal_emb is already in the shared latent space.

        info_dict:         dict with "pixels" observation history
        action_candidates: (B, S, T, action_dim)
        subgoal_emb:       (B, D) — target embedding from L2 plan

        returns: (B, S) cost
        """
        return self.l1_jepa.get_cost_from_emb(info_dict, action_candidates, subgoal_emb)
