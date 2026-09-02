#!/usr/bin/env python3
"""Comprehensive V4 Study: Rigorous Action-Conditioned JEPA Battle World Models.

This script implements ALL recommendations from pokemon-jepa-flaws.md:
1. True Latent JEPA: EMA target encoder with stop-gradient, latent VICReg collapse prevention (no raw HP MSE).
2. Action Conditioning: Predictor conditioned on both players' actions.
3. Non-Circular Probing: Fair evaluation of frozen latents against Supervised, Random, and Persistence baselines.
4. Naive / Persistence Baseline: Strict comparison against y_{t+1} = y_t, delta-HP metrics, and R^2 over persistence.
5. Positional & Slot Embeddings: Explicit learned side_embed, slot_embed, and active_embed before self-attention.
6. Canonical Slots: Fixed 0..5 slot indexing across entire battles, eliminating dynamic slot swapping corruption.
7. Split-First Vocabulary: Vocabulary fitted strictly on training partition with <unk> handling.
8. Tactical State: Full parsing of stat stage boosts (-6..+6), hazards, weather, and active status.
9. Teambuilder & Masked Species Head: Masked species prediction head and archetype completion testing.
10. Calibration & Fair Evaluation: Brier score, ECE, log loss, and honest metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .v4_baselines import compute_calibration_metrics, compute_hp_metrics, run_frozen_latent_probes
    from .v4_models import (
        MASK_TOKEN_ID,
        PAD_TOKEN_ID,
        UNK_TOKEN_ID,
        ActionConditionedPredictor,
        PositionalTransformerEncoder,
        Teambuilder,
        TrueJEPA,
    )
    from .v4_parser import (
        BattleState,
        PokemonVocab,
        make_v4_transitions,
        parse_v4_replay,
    )
except (ImportError, ValueError):
    from v4_baselines import compute_calibration_metrics, compute_hp_metrics, run_frozen_latent_probes
    from v4_models import (
        MASK_TOKEN_ID,
        PAD_TOKEN_ID,
        UNK_TOKEN_ID,
        ActionConditionedPredictor,
        PositionalTransformerEncoder,
        Teambuilder,
        TrueJEPA,
    )
    from v4_parser import (
        BattleState,
        PokemonVocab,
        make_v4_transitions,
        parse_v4_replay,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_REPLAYS = ROOT / "data" / "pokemon_jepa" / "pokemon_raw_replays_sample.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "v4"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class V4TransitionDataset(Dataset):
    def __init__(self, transitions: list[dict[str, Any]]) -> None:
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.transitions[idx]
        curr = item["curr"]
        nxt = item["next"]

        return {
            "curr_slots": torch.tensor(curr["slots"], dtype=torch.float32),        # (12, 9)
            "curr_field": torch.tensor(curr["field"], dtype=torch.float32),        # (9,)
            "curr_hps": torch.tensor(curr["hps"], dtype=torch.float32),            # (12,)
            "nxt_slots": torch.tensor(nxt["slots"], dtype=torch.float32),          # (12, 9)
            "nxt_field": torch.tensor(nxt["field"], dtype=torch.float32),          # (9,)
            "nxt_hps": torch.tensor(nxt["hps"], dtype=torch.float32),              # (12,)
            "p1_action": torch.tensor(item["p1_action"], dtype=torch.long),        # (3,)
            "p2_action": torch.tensor(item["p2_action"], dtype=torch.long),        # (3,)
            "delta_hp": torch.tensor(item["delta_hp"], dtype=torch.float32),        # (12,)
            "winner_p1": torch.tensor(item["winner_p1"], dtype=torch.float32),     # scalar
        }


def make_loader(transitions: list[dict[str, Any]], batch_size: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        V4TransitionDataset(transitions),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=device.type == "cuda",
        drop_last=shuffle and len(transitions) > batch_size,
    )


def train_true_jepa(
    jepa: TrueJEPA,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train True JEPA using latent VICReg loss and EMA target encoder updates."""
    optimizer = torch.optim.AdamW(
        list(jepa.online_encoder.parameters()) + list(jepa.predictor.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        jepa.train()
        total_loss = 0.0
        total_inv = 0.0
        total_var = 0.0
        total_cov = 0.0
        total_std = 0.0
        batches = 0

        for batch in loader:
            curr_slots = batch["curr_slots"].to(device, non_blocking=True)
            curr_field = batch["curr_field"].to(device, non_blocking=True)
            nxt_slots = batch["nxt_slots"].to(device, non_blocking=True)
            nxt_field = batch["nxt_field"].to(device, non_blocking=True)
            p1_act = batch["p1_action"].to(device, non_blocking=True)
            p2_act = batch["p2_action"].to(device, non_blocking=True)

            loss, metrics = jepa(curr_slots, curr_field, nxt_slots, nxt_field, p1_act, p2_act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(jepa.online_encoder.parameters()) + list(jepa.predictor.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            # Target encoder EMA update
            jepa.update_target_encoder()

            total_loss += metrics["total_jepa_loss"]
            total_inv += metrics["inv_loss"]
            total_var += metrics["var_loss"]
            total_cov += metrics["cov_loss"]
            total_std += metrics["latent_std_mean"]
            batches += 1

        denom = max(1, batches)
        epoch_stats = {
            "epoch": epoch + 1,
            "loss": total_loss / denom,
            "inv_loss": total_inv / denom,
            "var_loss": total_var / denom,
            "cov_loss": total_cov / denom,
            "std_mean": total_std / denom,
        }
        history.append(epoch_stats)

    return history


def train_supervised_baseline(
    encoder: PositionalTransformerEncoder,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train supervised baseline directly on winner BCE + direct HP regression."""
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            curr_slots = batch["curr_slots"].to(device, non_blocking=True)
            curr_field = batch["curr_field"].to(device, non_blocking=True)
            winner = batch["winner_p1"].to(device, non_blocking=True)

            out = encoder(curr_slots, curr_field)
            loss = bce(out["win_logit"], winner)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, batches)})

    return history


def train_masked_species_head(
    encoder: PositionalTransformerEncoder,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train species_head on masked slot tokens for teambuilder completion."""
    optimizer = torch.optim.AdamW(encoder.species_head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    for epoch in range(epochs):
        encoder.eval()
        encoder.species_head.train()
        for batch in loader:
            slots = batch["curr_slots"].clone().to(device)
            field = batch["curr_field"].to(device)
            batch_size = slots.shape[0]

            # Randomly pick 1 slot per sample (from P1 team slots 0..5) to mask
            mask_indices = torch.randint(0, 6, (batch_size,), device=device)
            true_species = torch.zeros(batch_size, dtype=torch.long, device=device)

            for b in range(batch_size):
                m_idx = mask_indices[b]
                true_species[b] = slots[b, m_idx, 0].long()
                slots[b, m_idx, 0] = float(MASK_TOKEN_ID)

            out = encoder(slots, field)
            # Gather masked slot token representations
            masked_tokens = []
            for b in range(batch_size):
                m_idx = mask_indices[b]
                masked_tokens.append(out["slot_tokens"][b, m_idx, :])
            masked_tokens = torch.stack(masked_tokens, dim=0)

            logits = encoder.species_head(masked_tokens)
            loss = ce(logits, true_species)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return float(loss.item())


@torch.no_grad()
def extract_latents(
    encoder: PositionalTransformerEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract frozen latents, current HPs, next HPs, and winner outcomes."""
    encoder.eval()
    latents, curr_hps, nxt_hps, winners = [], [], [], []
    for batch in loader:
        curr_slots = batch["curr_slots"].to(device, non_blocking=True)
        curr_field = batch["curr_field"].to(device, non_blocking=True)
        out = encoder(curr_slots, curr_field)
        latents.append(out["latent"].cpu().numpy())
        curr_hps.append(batch["curr_hps"].numpy())
        nxt_hps.append(batch["nxt_hps"].numpy())
        winners.append(batch["winner_p1"].numpy())

    return (
        np.concatenate(latents, axis=0),
        np.concatenate(curr_hps, axis=0),
        np.concatenate(nxt_hps, axis=0),
        np.concatenate(winners, axis=0),
    )


@torch.no_grad()
def evaluate_action_sensitivity(
    jepa: TrueJEPA,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Verify action conditioning: measure predicted latent distance when varying player actions."""
    jepa.eval()
    diff_norms = []
    cos_sims = []

    for batch in loader:
        curr_slots = batch["curr_slots"].to(device)
        curr_field = batch["curr_field"].to(device)
        p1_act = batch["p1_action"].to(device)
        p2_act = batch["p2_action"].to(device)

        s_t = jepa.online_encoder(curr_slots, curr_field)["latent"]
        # Actual action rollout
        pred_real = jepa.predictor(s_t, p1_act, p2_act)

        # Counterfactual: switch to alternate action (e.g. switch action kind: move <-> switch)
        p1_cf = p1_act.clone()
        p1_cf[:, 0] = torch.where(p1_act[:, 0] == 1, 2, 1)  # toggle move vs switch
        p1_cf[:, 1] = torch.where(p1_act[:, 0] == 1, 1, 10) # slot 1 vs move 10
        pred_cf = jepa.predictor(s_t, p1_cf, p2_act)

        diff = (pred_real - pred_cf).norm(dim=-1)
        norm_real = F.normalize(pred_real, dim=-1)
        norm_cf = F.normalize(pred_cf, dim=-1)
        cos_sim = (norm_real * norm_cf).sum(dim=-1)

        diff_norms.extend(diff.cpu().tolist())
        cos_sims.extend(cos_sim.cpu().tolist())

    return {
        "mean_action_sensitivity_norm": float(np.mean(diff_norms)),
        "mean_counterfactual_cosine_sim": float(np.mean(cos_sims)),
    }


@torch.no_grad()
def evaluate_teambuilder_accuracy(
    teambuilder: Teambuilder,
    transitions: list[dict[str, Any]],
    k_vals: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Evaluate top-1 and top-5 accuracy on held-out teams."""
    seen_teams = set()
    unique_teams: list[list[str]] = []

    for item in transitions:
        slots = item["curr"]["slots"][:6]  # P1 team
        team_species = []
        for s in slots:
            sp_id = int(s[0])
            name = teambuilder.id_to_species.get(sp_id, "unknown")
            if name not in ("unknown", "<pad>", "<unk>", "<mask_token>", "none"):
                team_species.append(name)
        if len(team_species) == 6:
            team_key = tuple(sorted(team_species))
            if team_key not in seen_teams:
                seen_teams.add(team_key)
                unique_teams.append(team_species)

    if not unique_teams:
        return {"teambuilder_top1": 0.0, "teambuilder_top5": 0.0, "teams_evaluated": 0}

    # Evaluate on up to 100 held-out teams
    eval_teams = unique_teams[:100]
    top1_hits = 0
    top5_hits = 0

    for team in eval_teams:
        target_species = team[5]
        context_species = team[:5]
        suggestions = teambuilder.suggest_sixth(context_species, top_k=5)
        suggested_names = [s[0].lower().replace(" ", "").replace("-", "") for s in suggestions]
        target_norm = target_species.lower().replace(" ", "").replace("-", "")

        if suggested_names and suggested_names[0] == target_norm:
            top1_hits += 1
        if any(name == target_norm for name in suggested_names[:5]):
            top5_hits += 1

    n = len(eval_teams)
    return {
        "teambuilder_top1": float(top1_hits / n),
        "teambuilder_top5": float(top5_hits / n),
        "teams_evaluated": n,
    }


def plot_v4_sample_efficiency(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    budgets = [r["budget"] for r in results]
    jepa_win = [r["jepa_win_acc"] * 100 for r in results]
    sup_win = [r["sup_win_acc"] * 100 for r in results]
    rand_win = [r["random_win_acc"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(budgets, jepa_win, marker="o", linewidth=2.5, color="#1f6feb", label="True JEPA Frozen Probe")
    ax.plot(budgets, sup_win, marker="s", linewidth=2.0, color="#d97706", linestyle="--", label="Supervised Baseline")
    ax.plot(budgets, rand_win, marker="^", linewidth=1.5, color="#94a3b8", linestyle=":", label="Random Untrained Encoder")
    ax.axhline(50.0, color="#cbd5e1", linestyle="--", label="Chance (50%)")

    ax.set_title("V4 Study: Downstream Winner Accuracy vs Training Replay Budget")
    ax.set_xlabel("Training replays")
    ax.set_ylabel("Held-out Winner Accuracy (%)")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()

    path = output_dir / "pokemon-jepa-v4-sample-efficiency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_v4_hp_vs_persistence(
    results: list[dict[str, Any]],
    persist_mse: float,
    output_dir: Path,
) -> Path:
    budgets = [r["budget"] for r in results]
    jepa_mse = [r["jepa_hp_mse"] for r in results]
    sup_mse = [r["sup_hp_mse"] for r in results]
    rand_mse = [r["random_hp_mse"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(budgets, jepa_mse, marker="o", linewidth=2.5, color="#1f6feb", label="True JEPA Frozen Probe")
    ax.plot(budgets, sup_mse, marker="s", linewidth=2.0, color="#d97706", linestyle="--", label="Supervised Baseline")
    ax.plot(budgets, rand_mse, marker="^", linewidth=1.5, color="#94a3b8", linestyle=":", label="Random Untrained Encoder")
    ax.axhline(persist_mse, color="#dc2626", linewidth=2.0, linestyle="-.", label=f"Persistence Baseline ({persist_mse:.4f})")

    ax.set_title("V4 Study: Next-HP Prediction vs Naive Persistence Baseline")
    ax.set_xlabel("Training replays")
    ax.set_ylabel("Held-out Next-HP MSE (lower is better)")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    path = output_dir / "pokemon-jepa-v4-hp-vs-persistence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_v4_r2_improvement(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    budgets = [str(int(r["budget"])) for r in results]
    r2_vals = [r["jepa_r2_over_persistence"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#1f6feb" if v >= 0 else "#ef4444" for v in r2_vals]
    bars = ax.bar(budgets, r2_vals, color=colors, alpha=0.88)
    ax.axhline(0, color="#64748b", linewidth=1.0)
    ax.set_title("V4 Study: R² Improvement Over Persistence Baseline")
    ax.set_xlabel("Training replays")
    ax.set_ylabel("R² over Persistence (%)")
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, r2_vals):
        y_pos = bar.get_height() + (0.5 if val >= 0 else -1.5)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()

    path = output_dir / "pokemon-jepa-v4-r2-improvement.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon JEPA V4 Study")
    parser.add_argument("--raw-replays-file", default=str(DEFAULT_RAW_REPLAYS))
    parser.add_argument("--max-replays", type=int, default=1000)
    parser.add_argument("--budgets", default="25,50,100,250,500,800")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"=== Running Pokemon JEPA V4 Study on {device} ===")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load raw replays
    raw_path = Path(args.raw_replays_file)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw replays file not found at {raw_path}")

    raw_replays = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_replays.append(json.loads(line))
            if len(raw_replays) >= args.max_replays:
                break

    print(f"Loaded {len(raw_replays)} raw replays.")

    # 2. Split FIRST to prevent vocabulary leakage (Flaw 2.3)
    rng = random.Random(args.seed)
    rng.shuffle(raw_replays)
    split_idx = int(0.8 * len(raw_replays))
    train_raw = raw_replays[:split_idx]
    test_raw = raw_replays[split_idx:]

    print(f"Split: {len(train_raw)} train replays, {len(test_raw)} test replays.")

    # 3. Fit vocabulary STRICTLY on training partition
    vocab = PokemonVocab()
    train_parsed = []
    for r in tqdm(train_raw, desc="Parsing Train Replays (fitting vocab)"):
        log_str = r.get("log", "")
        rows = parse_v4_replay(log_str, vocab, update_vocab=True)
        if len(rows) >= 2:
            train_parsed.append(rows)

    print(f"Species Vocab size: {len(vocab.species)} | Move Vocab size: {len(vocab.moves)}")

    # 4. Parse test replays WITHOUT updating vocab (unseen map to <unk>)
    test_parsed = []
    for r in tqdm(test_raw, desc="Parsing Test Replays (vocab frozen)"):
        log_str = r.get("log", "")
        rows = parse_v4_replay(log_str, vocab, update_vocab=False)
        if len(rows) >= 2:
            test_parsed.append(rows)

    train_transitions = make_v4_transitions(train_parsed)
    test_transitions = make_v4_transitions(test_parsed)

    print(f"Turn transitions: {len(train_transitions)} train, {len(test_transitions)} test.")

    test_loader = make_loader(test_transitions, args.batch_size, False, args.seed, device)

    # 5. Compute Persistence Baseline (Flaw 1.4)
    all_test_curr_hp = np.array([t["curr"]["hps"] for t in test_transitions])
    all_test_nxt_hp = np.array([t["next"]["hps"] for t in test_transitions])
    persist_metrics = compute_hp_metrics(all_test_nxt_hp, all_test_curr_hp, all_test_curr_hp)
    print("\n--- Naive Persistence Baseline (y_{{t+1}} = y_t) ---")
    print(f"Persistence HP MSE: {persist_metrics['persist_mse']:.5f}")
    print(f"Persistence HP MAE: {persist_metrics['persist_mae']:.5f}")
    print(f"Changed-slots zero-delta MSE: {persist_metrics['changed_slots_zero_baseline_mse']:.5f}")

    # 6. Evaluate Random (Untrained) Encoder baseline
    print("\n--- Evaluating Random Untrained Encoder Baseline ---")
    rand_encoder = PositionalTransformerEncoder(len(vocab.species)).to(device)
    train_sub_eval_loader = make_loader(train_transitions[:2000], args.batch_size, False, args.seed, device)
    z_rand_train, c_hp_train, n_hp_train, w_train = extract_latents(rand_encoder, train_sub_eval_loader, device)
    z_rand_test, c_hp_test, n_hp_test, w_test = extract_latents(rand_encoder, test_loader, device)
    rand_probe = run_frozen_latent_probes(
        z_rand_train, c_hp_train, n_hp_train, w_train,
        z_rand_test, c_hp_test, n_hp_test, w_test,
    )
    print(f"Random Encoder Win Probe: {rand_probe['win_acc'] * 100:.2f}% | HP MSE: {rand_probe['hp_mse']:.5f}")

    # 7. Parse budgets
    requested_budgets = sorted({int(b.strip()) for b in args.budgets.split(",") if b.strip()})
    budgets = [b for b in requested_budgets if b < len(train_parsed)]
    if len(train_parsed) not in budgets:
        budgets.append(len(train_parsed))
    print(f"\nEvaluating Replay Budgets: {budgets}")

    study_results: list[dict[str, Any]] = []

    for budget in budgets:
        print("\n=======================================================")
        print(f"=== Budget: {budget} replays ({len(train_parsed[:budget])} replays) ===")
        print(f"=======================================================")
        b_transitions = make_v4_transitions(train_parsed[:budget])
        b_train_loader = make_loader(b_transitions, args.batch_size, True, args.seed + budget, device)
        b_eval_loader = make_loader(b_transitions, args.batch_size, False, args.seed + budget, device)

        # A. Train True JEPA (Flaw 1.1, 1.2, 2.1)
        print(f"Training True JEPA (VICReg Latent Loss + EMA Target Encoder)...")
        jepa = TrueJEPA(len(vocab.species), len(vocab.moves)).to(device)
        jepa_hist = train_true_jepa(jepa, b_train_loader, args.epochs, args.lr, device)
        final_jepa = jepa_hist[-1]
        print(f"JEPA Final Epoch: Loss={final_jepa['loss']:.4f} | Inv={final_jepa['inv_loss']:.4f} | Var={final_jepa['var_loss']:.4f} | Std={final_jepa['std_mean']:.3f}")

        # Action sensitivity test (Flaw 1.2)
        act_sensitivity = evaluate_action_sensitivity(jepa, test_loader, device)
        print(f"Action Sensitivity: norm diff={act_sensitivity['mean_action_sensitivity_norm']:.4f} | counterfactual cos={act_sensitivity['mean_counterfactual_cosine_sim']:.4f}")

        # Extract frozen latents from True JEPA online encoder
        z_jepa_tr, c_hp_tr, n_hp_tr, w_tr = extract_latents(jepa.online_encoder, b_eval_loader, device)
        z_jepa_te, c_hp_te, n_hp_te, w_te = extract_latents(jepa.online_encoder, test_loader, device)
        jepa_probes = run_frozen_latent_probes(
            z_jepa_tr, c_hp_tr, n_hp_tr, w_tr,
            z_jepa_te, c_hp_te, n_hp_te, w_te,
        )

        # B. Train Supervised Baseline
        print(f"Training Supervised Baseline Encoder...")
        sup_encoder = PositionalTransformerEncoder(len(vocab.species)).to(device)
        sup_hist = train_supervised_baseline(sup_encoder, b_train_loader, args.epochs, args.lr, device)
        z_sup_tr, c_hp_str, n_hp_str, w_str = extract_latents(sup_encoder, b_eval_loader, device)
        z_sup_te, c_hp_ste, n_hp_ste, w_ste = extract_latents(sup_encoder, test_loader, device)
        sup_probes = run_frozen_latent_probes(
            z_sup_tr, c_hp_str, n_hp_str, w_str,
            z_sup_te, c_hp_ste, n_hp_ste, w_ste,
        )

        # C. Train Masked Species Head for Teambuilder (Flaw 3.2)
        train_masked_species_head(jepa.online_encoder, b_train_loader, epochs=4, lr=1e-3, device=device)
        teambuilder = Teambuilder(jepa.online_encoder, vocab, device)
        tb_metrics = evaluate_teambuilder_accuracy(teambuilder, test_transitions)
        print(f"Teambuilder Accuracy: Top-1={tb_metrics['teambuilder_top1'] * 100:.1f}% | Top-5={tb_metrics['teambuilder_top5'] * 100:.1f}%")

        budget_summary = {
            "budget": budget,
            "train_transitions": len(b_transitions),
            "jepa_final_loss": final_jepa["loss"],
            "jepa_latent_std": final_jepa["std_mean"],
            "action_sensitivity_norm": act_sensitivity["mean_action_sensitivity_norm"],
            "jepa_win_acc": jepa_probes["win_acc"],
            "jepa_brier_score": jepa_probes["brier_score"],
            "jepa_ece": jepa_probes["expected_calibration_error"],
            "jepa_hp_mse": jepa_probes["hp_mse"],
            "jepa_changed_slots_mse": jepa_probes["changed_slots_mse"],
            "jepa_r2_over_persistence": jepa_probes["r2_over_persistence"],
            "sup_win_acc": sup_probes["win_acc"],
            "sup_brier_score": sup_probes["brier_score"],
            "sup_ece": sup_probes["expected_calibration_error"],
            "sup_hp_mse": sup_probes["hp_mse"],
            "sup_r2_over_persistence": sup_probes["r2_over_persistence"],
            "random_win_acc": rand_probe["win_acc"],
            "random_hp_mse": rand_probe["hp_mse"],
            "teambuilder_top1": tb_metrics["teambuilder_top1"],
            "teambuilder_top5": tb_metrics["teambuilder_top5"],
        }
        study_results.append(budget_summary)

        print(
            f"--> JEPA Win: {jepa_probes['win_acc'] * 100:.1f}% (Brier: {jepa_probes['brier_score']:.3f}) | "
            f"SUP Win: {sup_probes['win_acc'] * 100:.1f}% | "
            f"JEPA HP MSE: {jepa_probes['hp_mse']:.5f} (R2: {jepa_probes['r2_over_persistence'] * 100:.1f}%)"
        )

        del jepa, sup_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 8. Interactive Teambuilder Demonstration (Flaw 3.2)
    print("\n--- Running Archetype Teambuilder Demonstrations ---")
    # Retrain on full training set to demonstrate archetype completions
    best_jepa = TrueJEPA(len(vocab.species), len(vocab.moves)).to(device)
    full_loader = make_loader(train_transitions, args.batch_size, True, args.seed, device)
    train_true_jepa(best_jepa, full_loader, args.epochs, args.lr, device)
    train_masked_species_head(best_jepa.online_encoder, full_loader, epochs=5, lr=1e-3, device=device)
    final_teambuilder = Teambuilder(best_jepa.online_encoder, vocab, device)

    archetype_demos = {}
    sample_teams = [
        ("Sand / Hazard Archetype", ["hippowdon", "excadrill", "corviknight", "slowking", "gliscor"]),
        ("PU Offense Archetype", ["delphox", "florges", "rotommow", "mudsdale", "skuntank"]),
        ("Special Pivot Balance", ["bellibolt", "appletun", "decidueyehisui", "articunogalar", "taurospaldeablaze"]),
    ]
    for label, team in sample_teams:
        suggestions = final_teambuilder.suggest_sixth(team, top_k=5)
        archetype_demos[label] = {
            "core_5": team,
            "top_5_suggestions": suggestions,
        }
        print(f"\nArchetype: {label}")
        print(f"Given 5: {', '.join(team)}")
        print("Top 5 recommended 6th slots:")
        for rank, (name, prob) in enumerate(suggestions, 1):
            print(f"  {rank}. {name} ({prob * 100:.1f}%)")

    # 9. Generate Figures
    plot1 = plot_v4_sample_efficiency(study_results, output_dir)
    plot2 = plot_v4_hp_vs_persistence(study_results, persist_metrics["persist_mse"], output_dir)
    plot3 = plot_v4_r2_improvement(study_results, output_dir)

    payload = {
        "title": "Pokemon JEPA V4 Study Results",
        "description": "Rigorous action-conditioned joint-embedding predictive architecture with VICReg collapse prevention.",
        "gaps_addressed": [
            "1.1 True Latent JEPA with EMA Target Encoder & VICReg loss",
            "1.2 Action conditioning with player move/switch/tera parsing",
            "1.3 Non-circular frozen latent linear probing",
            "1.4 Naive persistence baseline benchmark and delta-HP metrics",
            "2.1 Positional, side, slot, and active embeddings",
            "2.2 Canonical slot order fixing dynamic swapping corruption",
            "2.3 Split-first vocabulary fitting without test set leakage",
            "3.1 Stat stage boosts (-6..+6), field hazards, and weather",
            "3.2 Masked species prediction head and teambuilder module",
            "3.3 Calibration metrics (Brier score, ECE, log loss)",
        ],
        "persistence_baseline": persist_metrics,
        "random_encoder_baseline": rand_probe,
        "study_results": study_results,
        "archetype_teambuilder_demos": archetype_demos,
        "plots": {
            "sample_efficiency": str(plot1),
            "hp_vs_persistence": str(plot2),
            "r2_improvement": str(plot3),
        },
    }

    result_json_path = output_dir / "pokemon_jepa_v4_study.json"
    result_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=======================================================")
    print(f"V4 Study Complete! Saved study results to:")
    print(f"  JSON: {result_json_path}")
    print(f"  Plots: {plot1}")
    print(f"         {plot2}")
    print(f"         {plot3}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
