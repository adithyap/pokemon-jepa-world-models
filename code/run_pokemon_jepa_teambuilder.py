#!/usr/bin/env python3
"""Pokemon JEPA Teambuilder Edition with Checkpointing.

Features:
- Multi-token per Pokemon: [Species, Item, Move1, Move2, Move3, Move4]
- Masked Identity Prediction (MLM) for Teambuilding suggestions.
- JEPA Next-HP prediction.
- Winner auxiliary prediction.
- Best-model checkpoint saving.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SIDES = ("p1", "p2")
STATUS_TO_ID = {"": 0, "par": 1, "psn": 2, "tox": 3, "brn": 4, "frz": 5, "slp": 6}

def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())

def hp_fraction(detail: str) -> float:
    token = detail.split()[0]
    if token == "0" or "fnt" in detail: return 0.0
    if "/" not in token: return 1.0
    left, right = token.split("/", 1)
    try:
        denom = float(right.rstrip("%"))
        return max(0.0, min(1.0, float(left) / denom)) if denom > 0 else 1.0
    except ValueError: return 1.0

@dataclass
class PokemonState:
    species: str = "none"
    hp: float = 1.0
    status: str = ""
    item: str = "none"
    moves: list[str] = field(default_factory=lambda: ["none"] * 4)

@dataclass
class BattleState:
    turn: int = 0
    teams: dict[str, list[PokemonState]] = field(
        default_factory=lambda: {s: [PokemonState() for _ in range(6)] for s in SIDES}
    )

class PokemonVocab:
    def __init__(self):
        self.species = {"none": 0, "[MASK]": 1}
        self.items = {"none": 0, "[MASK]": 1}
        self.moves = {"none": 0, "[MASK]": 1}
        
    def add(self, kind: str, name: str):
        target = getattr(self, kind)
        name = normalized_name(name)
        if name not in target: target[name] = len(target)
            
    def get(self, kind: str, name: str) -> int:
        target = getattr(self, kind)
        return target.get(normalized_name(name), 0)

def parse_replay_teambuilder(log: str, vocab: PokemonVocab) -> list[dict[str, Any]]:
    state = BattleState()
    rows = []
    player_to_side = {}
    
    # Pre-populate teams from poke events
    poke_counts = {"p1": 0, "p2": 0}
    for line in log.splitlines():
        if not line.startswith("|"): continue
        parts = line.split("|")
        if len(parts) < 2: continue
        event = parts[1]
        if event == "poke" and len(parts) >= 4:
            side = parts[2][:2]
            species = parts[3].split(",")[0]
            vocab.add("species", species)
            if poke_counts[side] < 6:
                state.teams[side][poke_counts[side]].species = species
                poke_counts[side] += 1
        elif event == "move" and len(parts) >= 4: vocab.add("moves", parts[3])
        elif event == "player" and len(parts) >= 4: player_to_side[parts[3]] = parts[2]
        elif event == "-item" and len(parts) >= 4: vocab.add("items", parts[3])

    current_turn_data = None
    for line in log.splitlines():
        if not line.startswith("|"): continue
        parts = line.split("|")
        event = parts[1]
        if event == "turn":
            if current_turn_data: rows.append(current_turn_data)
            state.turn = int(parts[2])
            current_turn_data = {
                "turn": state.turn,
                "p1_team": [(vocab.get("species", p.species), vocab.get("items", p.item), [vocab.get("moves", m) for m in p.moves], p.hp, STATUS_TO_ID.get(p.status, 0)) for p in state.teams["p1"]],
                "p2_team": [(vocab.get("species", p.species), vocab.get("items", p.item), [vocab.get("moves", m) for m in p.moves], p.hp, STATUS_TO_ID.get(p.status, 0)) for p in state.teams["p2"]],
            }
        elif event in ("switch", "drag") and len(parts) >= 4:
            side, species = parts[2][:2], parts[3].split(",")[0]
            for p in state.teams[side]:
                if p.species == species or p.species == "none":
                    p.species, p.hp = species, (hp_fraction(parts[4]) if len(parts) >= 5 else p.hp)
                    idx = state.teams[side].index(p)
                    state.teams[side][0], state.teams[side][idx] = state.teams[side][idx], state.teams[side][0]
                    break
        elif event == "move" and len(parts) >= 4:
            side, move = parts[2][:2], parts[3]
            if move not in state.teams[side][0].moves and "none" in state.teams[side][0].moves:
                state.teams[side][0].moves[state.teams[side][0].moves.index("none")] = move
        elif event == "-item" and len(parts) >= 4: state.teams[parts[2][:2]][0].item = parts[3]
        elif event in ("-damage", "-heal") and len(parts) >= 4: state.teams[parts[2][:2]][0].hp = hp_fraction(parts[3])
        elif event == "win":
            winner_side = player_to_side.get(parts[2])
            for r in rows: r["winner_p1"] = 1 if winner_side == "p1" else 0
    return rows

class TeambuilderDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows) - 1
    def __getitem__(self, idx):
        curr, nxt = self.rows[idx], self.rows[idx+1]
        state = []
        for p in curr["p1_team"] + curr["p2_team"]: state.extend([p[0], p[1], *p[2], p[3], p[4]])
        return {"x": torch.tensor(state), "y_hp": torch.tensor([p[3] for p in nxt["p1_team"] + nxt["p2_team"]]), "winner": torch.tensor(curr.get("winner_p1", 0), dtype=torch.float32)}

class TeambuilderModel(nn.Module):
    def __init__(self, spec_size, item_size, move_size, latent_dim=256):
        super().__init__()
        self.spec_embed = nn.Embedding(spec_size, 64)
        self.item_embed = nn.Embedding(item_size, 32)
        self.move_embed = nn.Embedding(move_size, 32)
        self.status_embed = nn.Embedding(8, 8)
        self.proj = nn.Linear(64 + 32 + 128 + 1 + 8, 128)
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.latent_proj = nn.Linear(128 * 12, latent_dim)
        self.jepa_head = nn.Linear(latent_dim, 12)
        self.winner_head = nn.Linear(latent_dim, 1)
        self.mlm_species = nn.Linear(latent_dim, spec_size)
    def forward(self, x):
        batch_size = x.shape[0]; tokens = []
        for i in range(12):
            b = i * 8; s = self.spec_embed(x[:, b].long()); it = self.item_embed(x[:, b+1].long())
            m1 = self.move_embed(x[:, b+2].long()); m2 = self.move_embed(x[:, b+3].long())
            m3 = self.move_embed(x[:, b+4].long()); m4 = self.move_embed(x[:, b+5].long())
            h = x[:, b+6].unsqueeze(1); st = self.status_embed(x[:, b+7].long())
            tokens.append(self.proj(torch.cat([s, it, m1, m2, m3, m4, h, st], dim=1)).unsqueeze(1))
        z = self.latent_proj(self.transformer(torch.cat(tokens, dim=1)).view(batch_size, -1))
        return {"z": z, "hp": self.jepa_head(z), "win": self.winner_head(z).squeeze(1), "mlm": self.mlm_species(z)}

def train_variant(name, model, loader, epochs, device, use_jepa=True, use_win=True, path=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    mse, bce, ce = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train(); total_l = 0
        for batch in loader:
            x, y, w = batch["x"].to(device), batch["y_hp"].to(device), batch["winner"].to(device)
            x_m = x.clone(); mt = []
            for i in range(x.shape[0]):
                idx = random.randint(0, 5); mt.append(x[i, idx*8].item()); x_m[i, idx*8] = 1
            out = model(x_m); loss = ce(out["mlm"], torch.tensor(mt, device=device).long())
            if use_jepa: loss += mse(out["hp"], y)
            if use_win: loss += 0.5 * bce(out["win"], w)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total_l += loss.item()
        avg_l = total_l/len(loader)
        print(f"  {name} Epoch {epoch+1}, Loss: {avg_l:.4f}")
        if path and avg_l < best_loss:
            best_loss = avg_l; torch.save(model.state_dict(), path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-replays", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    ds = load_dataset("milkkarten/pokemon-showdown-replays-merged", streaming=True, split="train", trust_remote_code=True, token=False)
    vocab, all_rows = PokemonVocab(), []
    for ex in tqdm(ds, desc="Parsing", total=args.max_replays):
        if len(all_rows) // 20 >= args.max_replays: break
        rows = parse_replay_teambuilder(ex.get("log", ""), vocab)
        if rows: all_rows.extend(rows)

    dataset = TeambuilderDataset(all_rows)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train JEPA Champ
    model_j = TeambuilderModel(len(vocab.species), len(vocab.items), len(vocab.moves)).to(device)
    train_variant("JEPA-Full", model_j, loader, args.epochs, device, True, True, "pokemon_jepa_final.pt")
    
    # Train Supervised Challenger (Same Architecture, No JEPA objective)
    model_s = TeambuilderModel(len(vocab.species), len(vocab.items), len(vocab.moves)).to(device)
    train_variant("Supervised-only", model_s, loader, args.epochs, device, False, True, "pokemon_supervised_final.pt")

if __name__ == "__main__":
    main()
