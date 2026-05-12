#!/usr/bin/env python3
"""Generate latent-space and HP-delta maps for the Pokemon JEPA blog.

This script reuses the structured replay parser and model from
run_pokemon_jepa_v3_study.py, trains one JEPA encoder, then writes:

- pokemon-jepa-latent-space-map.png
- pokemon-jepa-hp-delta-map.png

Both files are saved to the project results directory. Pass --public-dir to
also copy them into a blog's public assets directory.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

import run_pokemon_jepa_v3_study as study


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results"


def copy_to_public(path: Path, public_dir: Path | None) -> None:
    if public_dir is None:
        return
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, public_dir / path.name)


def display_name(name: str) -> str:
    special = {
        "rotommow": "Rotom-Mow",
        "screamtail": "Scream Tail",
        "gastrodoneast": "Gastrodon-E",
        "taurospaldeaaqua": "Tauros-Aqua",
        "taurospaldeablaze": "Tauros-Blaze",
        "braviaryhisui": "Hisuian Braviary",
        "decidueyehisui": "Hisuian Decidueye",
        "articunogalar": "Galarian Articuno",
        "sandslashalola": "Alolan Sandslash",
    }
    if name in special:
        return special[name]
    if len(name) > 14:
        name = name[:13] + "."
    return name[:1].upper() + name[1:]


def load_split(args: argparse.Namespace, vocab: study.PokemonVocab) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    replay_rows = study.load_replay_rows(args, vocab)
    rng = random.Random(args.seed)
    rng.shuffle(replay_rows)
    split_idx = int(0.8 * len(replay_rows))
    return replay_rows[:split_idx], replay_rows[split_idx:]


def train_jepa(args: argparse.Namespace, vocab: study.PokemonVocab, train_replays: list[list[dict[str, Any]]],
               device: torch.device) -> tuple[study.TransformerEncoder, nn.Module, list[dict[str, Any]]]:
    train_examples = study.make_examples(train_replays[: args.train_replays])
    train_loader = study.make_loader(train_examples, args.batch_size, True, args.seed + 101, device)
    study.seed_everything(args.seed + 101)
    encoder = study.TransformerEncoder(len(vocab.species)).to(device)
    predictor = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 12)).to(device)
    study.train_model(encoder, predictor, train_loader, args.epochs, 0.5, device, args.lr)
    return encoder, predictor, train_examples


def collect_latent_points(encoder: study.TransformerEncoder, examples: list[dict[str, Any]],
                          device: torch.device, max_points: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(examples))
    if len(idxs) > max_points:
        idxs = rng.choice(idxs, size=max_points, replace=False)
    selected = [examples[int(idx)] for idx in idxs]
    loader = study.make_loader(selected, 256, False, seed, device)

    latents, winners, hp_swings, turns = [], [], [], []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            z, _ = encoder(x)
            latents.append(z.cpu().numpy())
            winners.append(batch["winner"].numpy())
            curr_hp = x[:, 1::3].detach().cpu().numpy()
            next_hp = batch["y_hp"].numpy()
            p1_delta = (next_hp[:, :6] - curr_hp[:, :6]).sum(axis=1)
            p2_delta = (next_hp[:, 6:] - curr_hp[:, 6:]).sum(axis=1)
            hp_swings.append(p1_delta - p2_delta)
    for item in selected:
        turns.append(item["curr"]["turn"])

    z = np.concatenate(latents)
    coords = PCA(n_components=2, random_state=seed).fit_transform(z)
    return {
        "coords": coords,
        "winner": np.concatenate(winners),
        "hp_swing": np.concatenate(hp_swings),
        "turn": np.asarray(turns),
    }


def plot_latent_space(points: dict[str, np.ndarray], output_dir: Path, public_dir: Path | None) -> Path:
    coords = points["coords"]
    winners = points["winner"]
    swing = points["hp_swing"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    colors = np.where(winners > 0.5, "#1f6feb", "#b42318")
    axes[0].scatter(coords[:, 0], coords[:, 1], c=colors, s=13, alpha=0.62, linewidths=0)
    axes[0].set_title("Held-out JEPA Latents by Winner")
    axes[0].text(0.02, 0.96, "blue = P1 win\nred = P2 win", transform=axes[0].transAxes,
                 va="top", fontsize=10, color="#475569")

    vmax = max(0.25, float(np.percentile(np.abs(swing), 95)))
    scatter = axes[1].scatter(coords[:, 0], coords[:, 1], c=swing, s=13, alpha=0.72,
                              cmap="RdBu", vmin=-vmax, vmax=vmax, linewidths=0)
    axes[1].set_title("Same Latents by Next-turn HP Swing")
    cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("P1 HP delta - P2 HP delta")

    for ax in axes:
        ax.set_xlabel("PC1 of latent z")
        ax.set_ylabel("PC2 of latent z")
        ax.grid(True, alpha=0.18)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Latent-space Map: What the JEPA Encoder Organizes", fontsize=16, fontweight="bold")
    path = output_dir / "pokemon-jepa-latent-space-map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    copy_to_public(path, public_dir)
    return path


def choose_hp_delta_example(examples: list[dict[str, Any]]) -> dict[str, Any]:
    def movement(item: dict[str, Any]) -> float:
        curr = np.asarray([p[1] for p in item["curr"]["p1_team"] + item["curr"]["p2_team"]])
        nxt = np.asarray([p[1] for p in item["next"]["p1_team"] + item["next"]["p2_team"]])
        nonzero = np.count_nonzero(np.abs(nxt - curr) > 0.02)
        return float(np.abs(nxt - curr).sum() + 0.15 * nonzero)

    return max(examples, key=movement)


def predict_example(encoder: study.TransformerEncoder, predictor: nn.Module, item: dict[str, Any],
                    device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curr = item["curr"]
    state: list[float] = []
    for pokemon in curr["p1_team"] + curr["p2_team"]:
        state.extend(pokemon)
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    encoder.eval()
    predictor.eval()
    with torch.no_grad():
        z, _ = encoder(x)
        pred_hp = predictor(z).cpu().numpy()[0]
    curr_hp = np.asarray([p[1] for p in curr["p1_team"] + curr["p2_team"]], dtype=np.float32)
    actual_hp = np.asarray([p[1] for p in item["next"]["p1_team"] + item["next"]["p2_team"]], dtype=np.float32)
    pred_hp = np.clip(pred_hp, 0.0, 1.0)
    return curr_hp, actual_hp, pred_hp


def plot_hp_delta_map(encoder: study.TransformerEncoder, predictor: nn.Module, examples: list[dict[str, Any]],
                      vocab: study.PokemonVocab, output_dir: Path, public_dir: Path | None,
                      device: torch.device) -> Path:
    item = choose_hp_delta_example(examples)
    curr_hp, actual_hp, pred_hp = predict_example(encoder, predictor, item, device)
    actual_delta = (actual_hp - curr_hp) * 100.0
    pred_delta = (pred_hp - curr_hp) * 100.0

    inv_species = {idx: name for name, idx in vocab.species.items()}
    labels = []
    for side, team in (("P1", item["curr"]["p1_team"]), ("P2", item["curr"]["p2_team"])):
        for idx, pokemon in enumerate(team):
            name = display_name(inv_species.get(pokemon[0], "none"))
            labels.append(f"{side}-{idx + 1}\n{name}")

    matrix = np.vstack([actual_delta, pred_delta])
    vmax = max(20.0, float(np.percentile(np.abs(matrix), 95)))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"HP-delta Map for One Held-out State Transition (turn {item['curr']['turn']})",
                 fontsize=15, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual ΔHP", "JEPA predicted ΔHP"])
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.axvline(5.5, color="#111827", linewidth=1.5)
    ax.text(2.5, -0.85, "P1 slots", ha="center", va="center", fontsize=10, color="#475569")
    ax.text(8.5, -0.85, "P2 slots", ha="center", va="center", fontsize=10, color="#475569")

    for row in range(2):
        for col in range(12):
            val = matrix[row, col]
            if abs(val) >= 2:
                ax.text(col, row, f"{val:+.0f}", ha="center", va="center",
                        color="#111827", fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("HP percentage points")
    fig.tight_layout()
    path = output_dir / "pokemon-jepa-hp-delta-map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    copy_to_public(path, public_dir)

    summary = {
        "turn": item["curr"]["turn"],
        "labels": labels,
        "current_hp": curr_hp.round(3).tolist(),
        "actual_delta_pct": actual_delta.round(1).tolist(),
        "predicted_delta_pct": pred_delta.round(1).tolist(),
    }
    (output_dir / "pokemon-jepa-hp-delta-map.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-replays", type=int, default=1000)
    parser.add_argument("--train-replays", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--raw-replays-file", default=str(study.DEFAULT_RAW_REPLAYS))
    parser.add_argument("--dataset", default="milkkarten/pokemon-showdown-replays-merged")
    parser.add_argument("--split", default="train")
    parser.add_argument("--format-filter", default="[gen 9] pu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--public-dir", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    study.seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    public_dir = Path(args.public_dir) if args.public_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    vocab = study.PokemonVocab()
    train_replays, test_replays = load_split(args, vocab)
    encoder, predictor, _ = train_jepa(args, vocab, train_replays, device)
    test_examples = study.make_examples(test_replays)

    points = collect_latent_points(encoder, test_examples, device, args.max_points, args.seed)
    latent_path = plot_latent_space(points, output_dir, public_dir)
    delta_path = plot_hp_delta_map(encoder, predictor, test_examples, vocab, output_dir, public_dir, device)
    print(f"Saved latent map to {latent_path}")
    print(f"Saved HP-delta map to {delta_path}")


if __name__ == "__main__":
    main()
