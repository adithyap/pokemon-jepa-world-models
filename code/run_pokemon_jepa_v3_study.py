#!/usr/bin/env python3
"""Comparative study: JEPA vs supervised Pokemon battle encoders.

This script now emits the richer data needed for the blog:

1. Winner accuracy over multiple replay budgets.
2. Frozen-latent HP dynamics probe MSE over the same budgets.
3. Full-study JSON plus PNG plots copied into the blog public directory.

The default path uses the cached raw replay sample from this experiment day.
Pass --refresh to stream a new sample from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SIDES = ("p1", "p2")
STATUS_TO_ID = {"": 0, "par": 1, "psn": 2, "tox": 3, "brn": 4, "frz": 5, "slp": 6}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_REPLAYS = ROOT / "data" / "pokemon_jepa" / "pokemon_raw_replays_sample.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results"


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def hp_fraction(detail: str) -> float:
    token = detail.split()[0]
    if token == "0" or "fnt" in detail:
        return 0.0
    if "/" not in token:
        return 1.0
    left, right = token.split("/", 1)
    try:
        denom = float(right.rstrip("%"))
        return max(0.0, min(1.0, float(left) / denom)) if denom > 0 else 1.0
    except ValueError:
        return 1.0


@dataclass
class PokemonState:
    species: str = "none"
    hp: float = 1.0
    status: str = ""


@dataclass
class BattleState:
    turn: int = 0
    teams: dict[str, list[PokemonState]] = field(
        default_factory=lambda: {side: [PokemonState() for _ in range(6)] for side in SIDES}
    )


class PokemonVocab:
    def __init__(self) -> None:
        self.species = {"none": 0}

    def add(self, name: str) -> None:
        name = normalized_name(name)
        if name not in self.species:
            self.species[name] = len(self.species)

    def get(self, name: str) -> int:
        return self.species.get(normalized_name(name), 0)


def snapshot_team(state: BattleState, vocab: PokemonVocab, side: str) -> list[tuple[int, float, int]]:
    return [
        (vocab.get(p.species), p.hp, STATUS_TO_ID.get(p.status, 0))
        for p in state.teams[side]
    ]


def parse_replay_structured(log: str, vocab: PokemonVocab) -> list[dict[str, Any]]:
    state = BattleState()
    rows: list[dict[str, Any]] = []
    player_to_side: dict[str, str] = {}
    poke_counts = {"p1": 0, "p2": 0}

    for line in log.splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        event = parts[1]
        if event == "poke" and len(parts) >= 4:
            side = parts[2][:2]
            species = parts[3].split(",")[0]
            vocab.add(species)
            if side in poke_counts and poke_counts[side] < 6:
                state.teams[side][poke_counts[side]].species = species
                poke_counts[side] += 1
        elif event == "player" and len(parts) >= 4:
            player_to_side[parts[3]] = parts[2]

    current_turn_data: dict[str, Any] | None = None
    for line in log.splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        event = parts[1]
        if event == "turn" and len(parts) >= 3:
            if current_turn_data:
                rows.append(current_turn_data)
            state.turn = int(parts[2])
            current_turn_data = {
                "turn": state.turn,
                "p1_team": snapshot_team(state, vocab, "p1"),
                "p2_team": snapshot_team(state, vocab, "p2"),
            }
        elif event in ("switch", "drag") and len(parts) >= 4:
            side, species = parts[2][:2], parts[3].split(",")[0]
            for idx, pokemon in enumerate(state.teams[side]):
                if normalized_name(pokemon.species) == normalized_name(species) or pokemon.species == "none":
                    pokemon.species = species
                    pokemon.hp = hp_fraction(parts[4]) if len(parts) >= 5 else pokemon.hp
                    state.teams[side][0], state.teams[side][idx] = state.teams[side][idx], state.teams[side][0]
                    break
        elif event in ("-damage", "-heal") and len(parts) >= 4:
            side = parts[2][:2]
            if side in state.teams:
                state.teams[side][0].hp = hp_fraction(parts[3])
        elif event == "faint" and len(parts) >= 3:
            side = parts[2][:2]
            if side in state.teams:
                state.teams[side][0].hp = 0.0
        elif event == "-status" and len(parts) >= 4:
            side = parts[2][:2]
            if side in state.teams:
                state.teams[side][0].status = parts[3]
        elif event == "-curestatus" and len(parts) >= 3:
            side = parts[2][:2]
            if side in state.teams:
                state.teams[side][0].status = ""
        elif event == "win" and len(parts) >= 3:
            winner_side = player_to_side.get(parts[2])
            for row in rows:
                row["winner_p1"] = 1 if winner_side == "p1" else 0

    return [row for row in rows if "winner_p1" in row]


def make_examples(replay_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for rows in replay_rows:
        for idx in range(len(rows) - 1):
            examples.append({"curr": rows[idx], "next": rows[idx + 1]})
    return examples


class PokemonDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        curr = self.examples[idx]["curr"]
        nxt = self.examples[idx]["next"]
        state: list[float] = []
        for pokemon in curr["p1_team"] + curr["p2_team"]:
            state.extend(pokemon)
        return {
            "x": torch.tensor(state, dtype=torch.float32),
            "y_hp": torch.tensor([p[1] for p in nxt["p1_team"] + nxt["p2_team"]], dtype=torch.float32),
            "winner": torch.tensor(curr["winner_p1"], dtype=torch.float32),
        }


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.species_embed = nn.Embedding(vocab_size, 64)
        self.status_embed = nn.Embedding(8, 8)
        self.proj = nn.Linear(64 + 1 + 8, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.latent_proj = nn.Linear(64 * 12, latent_dim)
        self.winner_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for idx in range(12):
            base = idx * 3
            species = self.species_embed(x[:, base].long())
            hp = x[:, base + 1].unsqueeze(1)
            status = self.status_embed(x[:, base + 2].long())
            tokens.append(self.proj(torch.cat([species, hp, status], dim=1)).unsqueeze(1))
        encoded = self.transformer(torch.cat(tokens, dim=1)).reshape(x.shape[0], -1)
        z = self.latent_proj(encoded)
        return z, self.winner_head(z).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(examples: list[dict[str, Any]], batch_size: int, shuffle: bool, seed: int,
                device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PokemonDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=device.type == "cuda",
    )


def train_model(encoder: TransformerEncoder, predictor: nn.Module | None, loader: DataLoader,
                epochs: int, win_weight: float, device: torch.device, lr: float) -> list[dict[str, float]]:
    params = list(encoder.parameters()) + (list(predictor.parameters()) if predictor else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        encoder.train()
        if predictor:
            predictor.train()
        total_loss = 0.0
        total_hp = 0.0
        total_win = 0.0
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y_hp = batch["y_hp"].to(device, non_blocking=True)
            winner = batch["winner"].to(device, non_blocking=True)
            z, win_logit = encoder(x)
            loss = torch.zeros((), device=device)
            hp_loss = torch.zeros((), device=device)
            if predictor:
                hp_loss = mse(predictor(z), y_hp)
                loss = loss + hp_loss
            win_loss = bce(win_logit, winner)
            loss = loss + win_weight * win_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_hp += float(hp_loss.item())
            total_win += float(win_loss.item())

        denom = max(1, len(loader))
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / denom,
            "hp_loss": total_hp / denom,
            "win_loss": total_win / denom,
        })
    return history


def collect_outputs(encoder: TransformerEncoder, predictor: nn.Module | None, loader: DataLoader,
                    device: torch.device) -> dict[str, np.ndarray | float | None]:
    encoder.eval()
    if predictor:
        predictor.eval()
    latents, winners, y_hps, logits, hp_preds = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            z, win_logit = encoder(x)
            latents.append(z.cpu().numpy())
            logits.append(win_logit.cpu().numpy())
            winners.append(batch["winner"].numpy())
            y_hps.append(batch["y_hp"].numpy())
            if predictor:
                hp_preds.append(predictor(z).cpu().numpy())

    y = np.concatenate(winners)
    logit = np.concatenate(logits)
    y_hp = np.concatenate(y_hps)
    result: dict[str, np.ndarray | float | None] = {
        "z": np.concatenate(latents),
        "winner": y,
        "y_hp": y_hp,
        "win_acc": accuracy_score(y, (logit >= 0).astype(np.float32)),
        "direct_hp_mse": None,
    }
    if hp_preds:
        result["direct_hp_mse"] = mean_squared_error(y_hp, np.concatenate(hp_preds))
    return result


def evaluate_models(enc_j: TransformerEncoder, pred_j: nn.Module, enc_s: TransformerEncoder,
                    train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> dict[str, float]:
    train_j = collect_outputs(enc_j, pred_j, train_loader, device)
    test_j = collect_outputs(enc_j, pred_j, test_loader, device)
    train_s = collect_outputs(enc_s, None, train_loader, device)
    test_s = collect_outputs(enc_s, None, test_loader, device)

    ridge_j = Ridge(alpha=1.0).fit(train_j["z"], train_j["y_hp"])
    ridge_s = Ridge(alpha=1.0).fit(train_s["z"], train_s["y_hp"])
    jepa_hp_probe = mean_squared_error(test_j["y_hp"], ridge_j.predict(test_j["z"]))
    supervised_hp_probe = mean_squared_error(test_s["y_hp"], ridge_s.predict(test_s["z"]))

    return {
        "jepa_win_acc": float(test_j["win_acc"]),
        "supervised_win_acc": float(test_s["win_acc"]),
        "jepa_direct_hp_mse": float(test_j["direct_hp_mse"]),
        "jepa_latent_hp_probe_mse": float(jepa_hp_probe),
        "supervised_latent_hp_probe_mse": float(supervised_hp_probe),
    }


def examples_to_arrays(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, winners, y_hps = [], [], []
    for item in examples:
        curr = item["curr"]
        nxt = item["next"]
        state: list[float] = []
        for pokemon in curr["p1_team"] + curr["p2_team"]:
            state.extend(pokemon)
        xs.append(state)
        winners.append(curr["winner_p1"])
        y_hps.append([p[1] for p in nxt["p1_team"] + nxt["p2_team"]])
    return np.asarray(xs, dtype=np.float32), np.asarray(winners, dtype=np.int64), np.asarray(y_hps, dtype=np.float32)


def baseline_features(x: np.ndarray) -> np.ndarray:
    hps = np.stack([x[:, idx * 3 + 1] for idx in range(12)], axis=1)
    p1_hp = hps[:, :6].sum(axis=1)
    p2_hp = hps[:, 6:].sum(axis=1)
    p1_alive = (hps[:, :6] > 0).sum(axis=1)
    p2_alive = (hps[:, 6:] > 0).sum(axis=1)
    return np.stack([p1_hp, p2_hp, p1_hp - p2_hp, p1_alive, p2_alive, p1_alive - p2_alive], axis=1)


def train_baselines(train_examples: list[dict[str, Any]], test_examples: list[dict[str, Any]]) -> dict[str, float]:
    x_train, y_train, _ = examples_to_arrays(train_examples)
    x_test, y_test, _ = examples_to_arrays(test_examples)

    hp_clf = LogisticRegression(max_iter=1000).fit(baseline_features(x_train), y_train)
    hp_acc = accuracy_score(y_test, hp_clf.predict(baseline_features(x_test)))

    gbdt = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=7,
    )
    gbdt.fit(x_train, y_train)
    gbdt_acc = accuracy_score(y_test, gbdt.predict(x_test))
    return {"hp_baseline_acc": float(hp_acc), "gbdt_acc": float(gbdt_acc)}


def load_replay_rows(args: argparse.Namespace, vocab: PokemonVocab) -> list[list[dict[str, Any]]]:
    replay_rows: list[list[dict[str, Any]]] = []
    raw_path = Path(args.raw_replays_file)

    if raw_path.exists() and not args.refresh:
        iterator = []
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    iterator.append(json.loads(line))
        source = str(raw_path)
    else:
        iterator = load_dataset(args.dataset, streaming=True, split=args.split, trust_remote_code=True, token=False)
        source = args.dataset

    progress = tqdm(iterator, desc=f"Parsing {source}", total=args.max_replays if not isinstance(iterator, list) else min(args.max_replays, len(iterator)))
    for example in progress:
        if len(replay_rows) >= args.max_replays:
            break
        fmt = str(example.get("format", "")).lower()
        if args.format_filter and args.format_filter.lower() not in fmt:
            continue
        rows = parse_replay_structured(example.get("log", ""), vocab)
        if len(rows) >= 2:
            replay_rows.append(rows)
    return replay_rows


def copy_to_public(path: Path, public_dir: Path | None) -> None:
    if public_dir is None:
        return
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, public_dir / path.name)


def plot_sample_efficiency(results: list[dict[str, float]], baselines: dict[str, float], output_dir: Path,
                           public_dir: Path | None) -> Path:
    budgets = [row["budget_replays"] for row in results]
    jepa = [row["jepa_win_acc"] * 100 for row in results]
    supervised = [row["supervised_win_acc"] * 100 for row in results]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(budgets, jepa, marker="o", linewidth=2.5, label="JEPA multi-task")
    ax.plot(budgets, supervised, marker="o", linewidth=2.5, label="Supervised-only")
    ax.axhline(baselines["gbdt_acc"] * 100, color="#64748b", linestyle="--", linewidth=1.8, label="GBDT full-data")
    ax.axhline(baselines["hp_baseline_acc"] * 100, color="#94a3b8", linestyle=":", linewidth=1.8, label="HP-only full-data")
    ax.set_title("Winner Accuracy vs Training Replay Budget")
    ax.set_xlabel("Training replays")
    ax.set_ylabel("Held-out winner accuracy (%)")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    path = output_dir / "pokemon-jepa-sample-efficiency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    copy_to_public(path, public_dir)
    return path


def plot_dynamics_probe(results: list[dict[str, float]], output_dir: Path, public_dir: Path | None) -> Path:
    budgets = [row["budget_replays"] for row in results]
    jepa_probe = [row["jepa_latent_hp_probe_mse"] for row in results]
    supervised_probe = [row["supervised_latent_hp_probe_mse"] for row in results]
    jepa_direct = [row["jepa_direct_hp_mse"] for row in results]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(budgets, jepa_probe, marker="o", linewidth=2.5, label="JEPA latent -> HP probe")
    ax.plot(budgets, supervised_probe, marker="o", linewidth=2.5, label="Supervised latent -> HP probe")
    ax.plot(budgets, jepa_direct, marker="s", linewidth=2.0, linestyle="--", label="JEPA trained HP head")
    ax.set_title("How Much Next-State Information Is in the Latent?")
    ax.set_xlabel("Training replays")
    ax.set_ylabel("Held-out next-HP MSE (lower is better)")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / "pokemon-jepa-dynamics-probe.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    copy_to_public(path, public_dir)
    return path


def plot_dynamics_advantage(results: list[dict[str, float]], output_dir: Path, public_dir: Path | None) -> Path:
    budgets = [row["budget_replays"] for row in results]
    reductions = [
        100.0 * (row["supervised_latent_hp_probe_mse"] - row["jepa_latent_hp_probe_mse"])
        / row["supervised_latent_hp_probe_mse"]
        for row in results
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar([str(int(budget)) for budget in budgets], reductions, color="#1f6feb", alpha=0.88)
    ax.set_title("JEPA Latents Retain More Next-State Information")
    ax.set_xlabel("Training replays")
    ax.set_ylabel("Next-HP probe MSE reduction vs supervised latent (%)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, reductions, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout()

    path = output_dir / "pokemon-jepa-dynamics-advantage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    copy_to_public(path, public_dir)
    return path


def parse_budgets(value: str, train_replay_count: int) -> list[int]:
    budgets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    return [budget for budget in budgets if 1 <= budget <= train_replay_count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-replays", type=int, default=1000)
    parser.add_argument("--budgets", default="25,50,100,250,500,800")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset", default="milkkarten/pokemon-showdown-replays-merged")
    parser.add_argument("--split", default="train")
    parser.add_argument("--format-filter", default="[gen 9] pu")
    parser.add_argument("--raw-replays-file", default=str(DEFAULT_RAW_REPLAYS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--public-dir", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    public_dir = Path(args.public_dir) if args.public_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = PokemonVocab()
    replay_rows = load_replay_rows(args, vocab)
    if len(replay_rows) < 10:
        raise RuntimeError("Not enough parsed replays to run the study.")

    rng = random.Random(args.seed)
    rng.shuffle(replay_rows)
    split_idx = int(0.8 * len(replay_rows))
    train_replays = replay_rows[:split_idx]
    test_replays = replay_rows[split_idx:]
    test_examples = make_examples(test_replays)
    full_train_examples = make_examples(train_replays)
    budgets = parse_budgets(args.budgets, len(train_replays))
    if not budgets:
        raise RuntimeError("No valid budgets after clipping to the training replay count.")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    test_loader = make_loader(test_examples, args.batch_size, False, args.seed, device)
    baselines = train_baselines(full_train_examples, test_examples)

    results: list[dict[str, float]] = []
    for budget in budgets:
        seed_everything(args.seed + budget)
        train_examples = make_examples(train_replays[:budget])
        train_loader = make_loader(train_examples, args.batch_size, True, args.seed + budget, device)
        train_eval_loader = make_loader(train_examples, args.batch_size, False, args.seed + budget, device)

        print(f"\n[Budget: {budget} training replays | {len(train_examples)} turn examples]")
        enc_j = TransformerEncoder(len(vocab.species)).to(device)
        pred_j = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 12)).to(device)
        hist_j = train_model(enc_j, pred_j, train_loader, args.epochs, 0.5, device, args.lr)

        enc_s = TransformerEncoder(len(vocab.species)).to(device)
        hist_s = train_model(enc_s, None, train_loader, args.epochs, 1.0, device, args.lr)

        metrics = evaluate_models(enc_j, pred_j, enc_s, train_eval_loader, test_loader, device)
        row = {
            "budget_replays": float(budget),
            "train_examples": float(len(train_examples)),
            **metrics,
            "jepa_final_loss": hist_j[-1]["loss"],
            "supervised_final_loss": hist_s[-1]["loss"],
        }
        results.append(row)
        print(
            f"JEPA win {row['jepa_win_acc']:.3f} | SUP win {row['supervised_win_acc']:.3f} | "
            f"JEPA HP probe {row['jepa_latent_hp_probe_mse']:.4f} | "
            f"SUP HP probe {row['supervised_latent_hp_probe_mse']:.4f}"
        )

        del enc_j, pred_j, enc_s
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot1 = plot_sample_efficiency(results, baselines, output_dir, public_dir)
    plot2 = plot_dynamics_probe(results, output_dir, public_dir)
    plot3 = plot_dynamics_advantage(results, output_dir, public_dir)

    payload = {
        "args": vars(args),
        "data": {
            "parsed_replays": len(replay_rows),
            "train_replays": len(train_replays),
            "test_replays": len(test_replays),
            "train_examples_full": len(full_train_examples),
            "test_examples": len(test_examples),
            "species_vocab": len(vocab.species),
        },
        "baselines": baselines,
        "budget_results": results,
        "plots": {
            "sample_efficiency": str(plot1),
            "dynamics_probe": str(plot2),
            "dynamics_advantage": str(plot3),
        },
    }
    result_path = output_dir / "pokemon_jepa_v3_study.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved study data to {result_path}")
    print(f"Saved plots to {plot1}, {plot2}, and {plot3}")


if __name__ == "__main__":
    main()
