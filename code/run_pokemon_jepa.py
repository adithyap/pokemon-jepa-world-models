#!/usr/bin/env python3
"""Local Pokemon Showdown JEPA-style pruning experiment.

Streams public Pokemon Showdown replays, parses compact turn-level state
features, trains simple baselines, then trains a small masked future-state
model on CUDA and evaluates frozen representation probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ACTION_TO_ID = {"none": 0, "move": 1, "switch": 2, "faint": 3}
SIDES = ("p1", "p2")
ROOT = Path(__file__).resolve().parents[1]


def stable_hash(value: str, buckets: int) -> int:
    if not value:
        return 0
    digest = hashlib.blake2b(value.lower().encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % buckets + 1


def player_side(token: str) -> str | None:
    if token.startswith("p1"):
        return "p1"
    if token.startswith("p2"):
        return "p2"
    return None


def normalized_format(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def clean_name(token: str) -> str:
    if ":" in token:
        return token.split(":", 1)[1].strip()
    return token.strip()


@dataclass
class SideState:
    active: str = ""
    active_hp: float = 1.0
    active_status: str = ""
    revealed: set[str] = field(default_factory=set)
    fainted: int = 0
    last_action: str = "none"
    last_move: str = ""


@dataclass
class BattleState:
    turn: int = 0
    weather: str = ""
    terrain: str = ""
    sides: dict[str, SideState] = field(
        default_factory=lambda: {side: SideState() for side in SIDES}
    )


def hp_fraction(detail: str) -> float:
    # Pokemon Showdown HP details commonly look like "83/100" or "0 fnt".
    token = detail.split()[0]
    if token == "0" or "fnt" in detail:
        return 0.0
    if "/" not in token:
        return 1.0
    left, right = token.split("/", 1)
    try:
        denom = float(right.rstrip("%"))
        if denom <= 0:
            return 1.0
        return max(0.0, min(1.0, float(left) / denom))
    except ValueError:
        return 1.0


def feature_vector(state: BattleState) -> list[float]:
    feats: list[float] = [min(state.turn, 200) / 200.0]
    feats.extend(
        [
            stable_hash(state.weather, 64) / 64.0,
            stable_hash(state.terrain, 64) / 64.0,
        ]
    )
    for side in SIDES:
        s = state.sides[side]
        action_id = ACTION_TO_ID.get(s.last_action, 0)
        feats.extend(
            [
                stable_hash(s.active, 512) / 512.0,
                s.active_hp,
                stable_hash(s.active_status, 32) / 32.0,
                min(len(s.revealed), 6) / 6.0,
                min(s.fainted, 6) / 6.0,
                action_id / 3.0,
                stable_hash(s.last_move, 512) / 512.0,
            ]
        )
    return feats


def parse_replay(log: str, winner: str | None) -> list[dict[str, Any]]:
    state = BattleState()
    rows: list[dict[str, Any]] = []
    winner_side: str | None = None
    player_to_side: dict[str, str] = {}

    for raw_line in log.splitlines():
        if not raw_line.startswith("|"):
            continue
        parts = raw_line.split("|")
        event = parts[1] if len(parts) > 1 else ""

        if event == "player" and len(parts) >= 4:
            player_to_side[parts[3]] = parts[2]
        elif event == "turn" and len(parts) >= 3:
            if state.turn > 0:
                rows.append({"turn": state.turn, "x": feature_vector(state)})
                for side in SIDES:
                    state.sides[side].last_action = "none"
                    state.sides[side].last_move = ""
            try:
                state.turn = int(parts[2])
            except ValueError:
                state.turn += 1
        elif event in {"switch", "drag"} and len(parts) >= 4:
            side = player_side(parts[2])
            if side is not None:
                species = parts[3].split(",", 1)[0].strip()
                state.sides[side].active = species
                state.sides[side].active_hp = hp_fraction(parts[4]) if len(parts) >= 5 else 1.0
                state.sides[side].revealed.add(species)
                state.sides[side].last_action = "switch"
        elif event == "move" and len(parts) >= 4:
            side = player_side(parts[2])
            if side is not None:
                state.sides[side].active = clean_name(parts[2])
                state.sides[side].revealed.add(state.sides[side].active)
                state.sides[side].last_action = "move"
                state.sides[side].last_move = parts[3]
        elif event in {"-damage", "-heal"} and len(parts) >= 4:
            side = player_side(parts[2])
            if side is not None:
                state.sides[side].active_hp = hp_fraction(parts[3])
        elif event == "-status" and len(parts) >= 4:
            side = player_side(parts[2])
            if side is not None:
                state.sides[side].active_status = parts[3]
        elif event == "-curestatus" and len(parts) >= 3:
            side = player_side(parts[2])
            if side is not None:
                state.sides[side].active_status = ""
        elif event == "faint" and len(parts) >= 3:
            side = player_side(parts[2])
            if side is not None:
                state.sides[side].active_hp = 0.0
                state.sides[side].fainted = min(6, state.sides[side].fainted + 1)
                state.sides[side].last_action = "faint"
        elif event == "-weather" and len(parts) >= 3:
            state.weather = "" if parts[2] == "none" else parts[2]
        elif event in {"-fieldstart", "-fieldend"} and len(parts) >= 3:
            state.terrain = parts[2] if event == "-fieldstart" else ""
        elif event == "win" and len(parts) >= 3:
            winner_side = player_to_side.get(parts[2])

    if state.turn > 0:
        rows.append({"turn": state.turn, "x": feature_vector(state)})

    if winner_side is None and winner:
        winner_side = player_to_side.get(winner, winner if winner in SIDES else None)
    if winner_side not in {"p1", "p2"}:
        return []
    for i, row in enumerate(rows):
        row["winner_p1"] = 1 if winner_side == "p1" else 0
        if i + 1 < len(rows):
            next_x = rows[i + 1]["x"]
            row["future_x"] = next_x
            row["next_p1_action"] = int(round(next_x[8] * 3.0))
        else:
            row["future_x"] = row["x"]
            row["next_p1_action"] = 0
    return rows[:-1]


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows_path = data_dir / "pokemon_turn_rows.jsonl"
    raw_path = data_dir / "pokemon_raw_replays_sample.jsonl"

    if rows_path.exists() and not args.refresh:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        return rows, {"source": "cache", "rows": len(rows), "path": str(rows_path)}

    load_kwargs: dict[str, Any] = {
        "split": args.split,
        "streaming": True,
        "cache_dir": str(data_dir / "hf_cache"),
    }
    if args.data_file:
        load_kwargs["data_files"] = args.data_file
    ds = load_dataset(args.dataset, **load_kwargs)
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = defaultdict(int)
    stats["source"] = args.dataset
    stats["format_filter"] = args.format_filter

    with rows_path.open("w", encoding="utf-8") as rows_f, raw_path.open("w", encoding="utf-8") as raw_f:
        stream = tqdm(ds, desc="stream replays", total=args.max_replays)
        for example in stream:
            fmt = str(example.get("format", example.get("format_id", ""))).lower()
            if args.format_filter and normalized_format(args.format_filter) not in normalized_format(fmt):
                continue
            log = example.get("log") or example.get("battle_log") or example.get("text") or ""
            if not log:
                continue
            winner = None
            metadata = example.get("metadata")
            if isinstance(metadata, dict):
                winner = metadata.get("winner")
            winner = winner or example.get("winner")
            parsed = parse_replay(str(log), winner)
            stats["seen_replays"] += 1
            if parsed:
                stats["parsed_replays"] += 1
                raw_f.write(json.dumps({"format": fmt, "log": log[:20000], "winner": winner}) + "\n")
                for row in parsed:
                    rows.append(row)
                    rows_f.write(json.dumps(row) + "\n")
            if stats["parsed_replays"] >= args.max_replays:
                break
            stream.set_postfix(parsed=stats["parsed_replays"], rows=len(rows))

    stats["rows"] = len(rows)
    stats["rows_path"] = str(rows_path)
    stats["raw_path"] = str(raw_path)
    return rows, dict(stats)


class MaskedFutureModel(nn.Module):
    def __init__(self, dim: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent + dim + dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, future_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.predictor(torch.cat([z, future_masked, mask], dim=1))


def masked_future_train(
    x_train: np.ndarray,
    future_train: np.ndarray,
    x_test: np.ndarray,
    future_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[MaskedFutureModel, dict[str, float], np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    model = MaskedFutureModel(x_train.shape[1], args.hidden_dim, args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(future_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, fb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            mask = (torch.rand_like(fb) < args.mask_prob).float()
            masked_future = fb * (1.0 - mask)
            pred = model(xb, masked_future, mask)
            loss = loss_fn(pred * mask, fb * mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch={epoch + 1} train_masked_mse={np.mean(losses):.6f}", flush=True)

    def eval_loss(x: np.ndarray, future: np.ndarray) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(x), args.batch_size):
                xb = torch.tensor(x[start : start + args.batch_size], dtype=torch.float32, device=device)
                fb = torch.tensor(future[start : start + args.batch_size], dtype=torch.float32, device=device)
                mask = torch.ones_like(fb)
                pred = model(xb, torch.zeros_like(fb), mask)
                losses.append(loss_fn(pred, fb).item())
        return float(np.mean(losses))

    def encode(x: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), args.batch_size):
                xb = torch.tensor(x[start : start + args.batch_size], dtype=torch.float32, device=device)
                outputs.append(model.encoder(xb).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    metrics = {
        "test_full_future_mse": eval_loss(x_test, future_test),
        "device": str(device),
    }
    return model, metrics, encode(x_train), encode(x_test)


def fit_winner_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        return {"auc": float("nan"), "log_loss": float("nan"), "accuracy": float("nan")}
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, probs)),
        "log_loss": float(log_loss(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
    }


def fit_action_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    if len(set(y_train.tolist())) < 2:
        return {"accuracy": float("nan"), "top2_accuracy": float("nan")}
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)
    preds = clf.predict(x_test)
    out = {"accuracy": float(accuracy_score(y_test, preds))}
    try:
        out["top2_accuracy"] = float(top_k_accuracy_score(y_test, probs, k=min(2, probs.shape[1]), labels=clf.classes_))
    except ValueError:
        out["top2_accuracy"] = float("nan")
    out["classes"] = len(classes)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="milkkarten/pokemon-showdown-replays-merged")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-file", default="data/part-00000.parquet")
    parser.add_argument("--format-filter", default="gen9randombattle")
    parser.add_argument("--max-replays", type=int, default=1000)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "pokemon_jepa"))
    parser.add_argument("--output", default=str(ROOT / "results" / "pokemon_jepa_results.json"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--mask-prob", type=float, default=0.45)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rows, data_stats = load_rows(args)
    if len(rows) < 200:
        raise SystemExit(f"Need at least 200 parsed rows, got {len(rows)} from {data_stats}")

    x = np.array([row["x"] for row in rows], dtype=np.float32)
    future = np.array([row["future_x"] for row in rows], dtype=np.float32)
    winner = np.array([row["winner_p1"] for row in rows], dtype=np.int64)
    next_action = np.array([row["next_p1_action"] for row in rows], dtype=np.int64)

    idx = np.arange(len(x))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=winner)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    future_scaler = StandardScaler()
    future_train = future_scaler.fit_transform(future[train_idx])
    future_test = future_scaler.transform(future[test_idx])

    raw_winner = fit_winner_probe(x_train, winner[train_idx], x_test, winner[test_idx])
    raw_action = fit_action_probe(x_train, next_action[train_idx], x_test, next_action[test_idx])
    _, jepa_metrics, z_train, z_test = masked_future_train(x_train, future_train, x_test, future_test, args)
    z_winner = fit_winner_probe(z_train, winner[train_idx], z_test, winner[test_idx])
    z_action = fit_action_probe(z_train, next_action[train_idx], z_test, next_action[test_idx])

    output = {
        "args": vars(args),
        "data": data_stats,
        "n_rows": int(len(rows)),
        "n_features": int(x.shape[1]),
        "winner_balance_p1": float(winner.mean()),
        "next_action_counts": {str(k): int(v) for k, v in zip(*np.unique(next_action, return_counts=True))},
        "raw_feature_winner_probe": raw_winner,
        "raw_feature_next_action_probe": raw_action,
        "masked_future_model": jepa_metrics,
        "jepa_embedding_winner_probe": z_winner,
        "jepa_embedding_next_action_probe": z_action,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
