"""PyTorch model architectures for V4 JEPA World Models study.

Addresses the following flaws identified in pokemon-jepa-flaws.md:
- 1.1 True Latent JEPA: EMA target encoder with stop-gradient, latent space predictor, VICReg collapse prevention.
- 1.2 Action Conditioning: Transition predictor conditioned on both players' actions.
- 2.1 Positional & Slot Embeddings: Explicit learned side_embed, slot_embed, active_embed before self-attention.
- 3.2 Masked Species Head & Teambuilder: Slot-level species classification head for filling missing team slots.
"""

from __future__ import annotations

import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any

try:
    from .v4_parser import MASK_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID
except (ImportError, ValueError):
    from v4_parser import MASK_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID


class PositionalTransformerEncoder(nn.Module):
    """Transformer encoder with explicit slot, side, active, and field embeddings."""

    def __init__(
        self,
        species_vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        latent_dim: int = 128,
        status_dim: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.species_vocab_size = species_vocab_size

        # Slot feature embeddings
        self.species_embed = nn.Embedding(species_vocab_size, 64)
        self.status_embed = nn.Embedding(8, status_dim)
        
        # Continuous boosts (5) + hp (1) + species (64) + status (8) + active_flag (1)
        slot_raw_dim = 64 + 1 + status_dim + 1 + 5
        self.slot_mlp = nn.Sequential(
            nn.Linear(slot_raw_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Explicit Positional / Board Embeddings (Flaw 2.1 fix)
        # Side ID: 0 (P1), 1 (P2)
        self.side_embed = nn.Embedding(2, d_model)
        # Slot ID: 0..5 for each side
        self.slot_embed = nn.Embedding(6, d_model)
        # Active status: 0 (bench), 1 (active battler)
        self.active_embed = nn.Embedding(2, d_model)

        # Field token: weather (5 classes) + 8 hazard values
        self.weather_embed = nn.Embedding(5, 16)
        self.field_mlp = nn.Sequential(
            nn.Linear(16 + 8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Pooling to latent representation
        # 12 slot tokens + 1 field token = 13 tokens
        self.latent_proj = nn.Sequential(
            nn.Linear(13 * d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Masked Species Prediction Head (Flaw 3.2 fix)
        # Maps individual slot token representation -> species vocabulary logits
        self.species_head = nn.Linear(d_model, species_vocab_size)

        # Winner Head (for supervised baseline or probed evaluation)
        self.winner_head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        slots: torch.Tensor,
        field: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.
        slots: (B, 12, 9)
               dim 0: species_id (long)
               dim 1: hp (float)
               dim 2: status_id (long)
               dim 3: is_active (long: 0 or 1)
               dim 4..8: boosts (5 floats)
        field: (B, 9)
               dim 0: weather_id (long)
               dim 1..8: hazards (8 floats)
        """
        batch_size = slots.shape[0]
        device = slots.device

        # 1. Build slot tokens
        slot_tokens = []
        for i in range(12):
            side_id = 0 if i < 6 else 1
            slot_id = i % 6
            raw_slot = slots[:, i, :]

            spec_id = raw_slot[:, 0].long().clamp(0, self.species_vocab_size - 1)
            hp = raw_slot[:, 1:2]
            stat_id = raw_slot[:, 2].long().clamp(0, 7)
            active_id = raw_slot[:, 3].long().clamp(0, 1)
            boosts = raw_slot[:, 4:9]

            spec_vec = self.species_embed(spec_id)
            stat_vec = self.status_embed(stat_id)
            raw_cat = torch.cat([spec_vec, hp, stat_vec, active_id.unsqueeze(1).float(), boosts], dim=-1)
            base_tok = self.slot_mlp(raw_cat)

            # Add explicit side, slot, and active embeddings before self-attention!
            side_tensor = torch.tensor(side_id, dtype=torch.long, device=device)
            slot_tensor = torch.tensor(slot_id, dtype=torch.long, device=device)
            pos_tok = (
                base_tok
                + self.side_embed(side_tensor).unsqueeze(0)
                + self.slot_embed(slot_tensor).unsqueeze(0)
                + self.active_embed(active_id)
            )
            slot_tokens.append(pos_tok.unsqueeze(1))

        # 2. Build field token
        weather_id = field[:, 0].long().clamp(0, 4)
        hazards = field[:, 1:9]
        weather_vec = self.weather_embed(weather_id)
        field_tok = self.field_mlp(torch.cat([weather_vec, hazards], dim=-1)).unsqueeze(1)

        # 3. Concatenate all 13 tokens: [field, slot_0 .. slot_11]
        all_tokens = torch.cat([field_tok] + slot_tokens, dim=1)  # (B, 13, d_model)

        # 4. Self-attention over board tokens
        encoded_tokens = self.transformer(all_tokens)  # (B, 13, d_model)

        # 5. Global latent representation
        flattened = encoded_tokens.reshape(batch_size, -1)
        z = self.latent_proj(flattened)
        win_logit = self.winner_head(z).squeeze(-1)

        return {
            "latent": z,
            "tokens": encoded_tokens,
            "slot_tokens": encoded_tokens[:, 1:, :],  # (B, 12, d_model)
            "field_token": encoded_tokens[:, 0, :],   # (B, d_model)
            "win_logit": win_logit,
        }


class ActionConditionedPredictor(nn.Module):
    """Predictor P(s_t, a_t) mapping context latent and actions to next latent representation."""

    def __init__(
        self,
        latent_dim: int = 128,
        move_vocab_size: int = 500,
        action_dim: int = 64,
    ) -> None:
        super().__init__()
        self.move_vocab_size = move_vocab_size
        self.action_kind_embed = nn.Embedding(3, 16)      # 0=none, 1=move, 2=switch
        self.move_embed = nn.Embedding(move_vocab_size, 32)
        self.switch_embed = nn.Embedding(6, 32)
        self.tera_embed = nn.Embedding(2, 8)

        # Single player action encoder
        player_act_raw_dim = 16 + 32 + 8
        self.player_act_mlp = nn.Sequential(
            nn.Linear(player_act_raw_dim, action_dim // 2),
            nn.GELU(),
        )

        # Predictor MLP: [s_t, a_p1, a_p2] -> s_hat_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Encode single player action tensor (B, 3): [kind, target, tera]."""
        kind = action[:, 0].long().clamp(0, 2)
        target = action[:, 1].long()
        tera = action[:, 2].long().clamp(0, 1)

        kind_vec = self.action_kind_embed(kind)
        tera_vec = self.tera_embed(tera)

        # Target can be a move_id or a switch slot
        move_mask = (kind == 1).unsqueeze(-1)
        switch_mask = (kind == 2).unsqueeze(-1)

        target_clamped_move = target.clamp(0, self.move_vocab_size - 1)
        target_clamped_switch = target.clamp(0, 5)

        move_vec = self.move_embed(target_clamped_move)
        switch_vec = self.switch_embed(target_clamped_switch)
        target_vec = torch.where(move_mask, move_vec, torch.where(switch_mask, switch_vec, torch.zeros_like(move_vec)))

        raw = torch.cat([kind_vec, target_vec, tera_vec], dim=-1)
        return self.player_act_mlp(raw)

    def forward(
        self,
        context_latent: torch.Tensor,
        p1_action: torch.Tensor,
        p2_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next latent state given context latent and both actions."""
        a1 = self.encode_action(p1_action)
        a2 = self.encode_action(p2_action)
        actions = torch.cat([a1, a2], dim=-1)
        return self.predictor(torch.cat([context_latent, actions], dim=-1))


class TrueJEPA(nn.Module):
    """Mathematically rigorous Joint-Embedding Predictive Architecture (JEPA).

    - Online context encoder E_theta(x_t) -> s_t
    - Target encoder E_xi(x_{t+1}) -> s_{t+1} with EMA update and stop-gradient
    - Action-conditioned latent predictor P_psi(s_t, a_t) -> s_hat_{t+1}
    - VICReg collapse prevention (variance + covariance regularization)
    """

    def __init__(
        self,
        species_vocab_size: int,
        move_vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        latent_dim: int = 128,
        ema_decay: float = 0.996,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.var_weight = var_weight
        self.cov_weight = cov_weight

        # Online Context Encoder
        self.online_encoder = PositionalTransformerEncoder(
            species_vocab_size=species_vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            latent_dim=latent_dim,
        )

        # Target Encoder (EMA copy with stop-gradient)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Action-Conditioned Latent Predictor
        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            move_vocab_size=move_vocab_size,
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Exponential Moving Average (EMA) update: xi <- tau * xi + (1 - tau) * theta."""
        tau = self.ema_decay
        for p_online, p_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    def vicreg_loss(
        self,
        predicted_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute True JEPA loss: Invariance + Variance Regularization + Covariance Regularization."""
        # 1. Invariance / Cosine or Smooth L1 loss in latent space
        norm_pred = F.normalize(predicted_latent, dim=-1)
        norm_target = F.normalize(target_latent, dim=-1)
        inv_loss = (1.0 - (norm_pred * norm_target).sum(dim=-1)).mean()

        # 2. Variance loss (prevents representation collapse)
        # Keeps standard deviation along each latent dimension >= 1.0
        std_pred = torch.sqrt(predicted_latent.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std_pred))

        # 3. Covariance loss (prevents dimensional collapse / redundancy)
        # Off-diagonal elements of covariance matrix penalized to 0
        batch_size = predicted_latent.shape[0]
        if batch_size > 1:
            z_centered = predicted_latent - predicted_latent.mean(dim=0)
            cov = (z_centered.T @ z_centered) / (batch_size - 1)
            # Zero out diagonal
            off_diag = cov - torch.diag(torch.diagonal(cov))
            cov_loss = (off_diag ** 2).sum() / self.latent_dim
        else:
            cov_loss = torch.zeros((), device=predicted_latent.device)

        total_loss = inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        metrics = {
            "inv_loss": float(inv_loss.item()),
            "var_loss": float(var_loss.item()),
            "cov_loss": float(cov_loss.item()),
            "latent_std_mean": float(std_pred.mean().item()),
            "total_jepa_loss": float(total_loss.item()),
        }
        return total_loss, metrics

    def forward(
        self,
        curr_slots: torch.Tensor,
        curr_field: torch.Tensor,
        nxt_slots: torch.Tensor,
        nxt_field: torch.Tensor,
        p1_action: torch.Tensor,
        p2_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Context encoding through online encoder
        context_out = self.online_encoder(curr_slots, curr_field)
        s_t = context_out["latent"]

        # Target encoding through target encoder with stop-gradient
        with torch.no_grad():
            target_out = self.target_encoder(nxt_slots, nxt_field)
            s_tp1 = target_out["latent"].detach()

        # Action-conditioned latent prediction
        s_hat_tp1 = self.predictor(s_t, p1_action, p2_action)

        loss, metrics = self.vicreg_loss(s_hat_tp1, s_tp1)
        return loss, metrics


class Teambuilder:
    """Masked species teambuilder (Flaw 3.2 fix).

    Given 5 Pokemon on a team, masks the 6th slot and produces ranked
    species recommendations with softmax probabilities from the trained encoder.
    """

    def __init__(
        self,
        encoder: PositionalTransformerEncoder,
        vocab: Any,
        device: torch.device,
    ) -> None:
        self.encoder = encoder
        self.vocab = vocab
        self.device = device
        # Invert species vocab
        self.id_to_species = {idx: name for name, idx in vocab.species.items()}

    def suggest_sixth(
        self,
        team_species: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Suggest 6th slot given 5 species names."""
        self.encoder.eval()

        # Build mock 12-slot board: P1 team has 5 species + 1 masked; P2 has none
        slots_data: list[list[float]] = []
        existing_norm = set()

        for idx in range(6):
            if idx < len(team_species):
                sp = team_species[idx]
                existing_norm.add(sp.lower().replace(" ", "").replace("-", ""))
                sp_id = float(self.vocab.get_species(sp))
            else:
                sp_id = float(MASK_TOKEN_ID)
            # species, hp=1.0, status=0, active=(idx==0), boosts=0
            slots_data.append([sp_id, 1.0, 0.0, 1.0 if idx == 0 else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # P2 slots empty
        for _ in range(6):
            slots_data.append([float(PAD_TOKEN_ID), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        field_data = [0.0] * 9  # no weather, no hazards

        slots_tensor = torch.tensor([slots_data], dtype=torch.float32, device=self.device)
        field_tensor = torch.tensor([field_data], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.encoder(slots_tensor, field_tensor)
            # Slot token 5 is the masked slot (0-indexed)
            masked_slot_tok = out["slot_tokens"][:, 5, :]  # (1, d_model)
            logits = self.encoder.species_head(masked_slot_tok)[0]  # (vocab_size,)

            # Mask special tokens and already selected species
            logits[PAD_TOKEN_ID] = -1e9
            logits[UNK_TOKEN_ID] = -1e9
            logits[MASK_TOKEN_ID] = -1e9

            for name, sp_id in self.vocab.species.items():
                norm = name.lower().replace(" ", "").replace("-", "")
                if norm in existing_norm:
                    logits[sp_id] = -1e9

            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.vocab.species)))

            results = []
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                if prob > 0:
                    results.append((self.id_to_species.get(idx, "unknown"), float(prob)))
            return results
