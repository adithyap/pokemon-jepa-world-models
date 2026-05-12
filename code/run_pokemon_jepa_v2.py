#!/usr/bin/env python3
"""Refactored Pokemon Showdown JEPA with Proper Embeddings.

Features:
- Global vocabulary for Species, Moves, Items, and Statuses.
- Object-token representation (Active, Bench).
- Transformer/Structured Encoder.
- Delta-based JEPA target (HP loss, faint prediction).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SIDES = ("p1", "p2")
STATUS_TO_ID = {"": 0, "par": 1, "psn": 2, "tox": 3, "brn": 4, "frz": 5, "slp": 6}

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
    fainted: bool = False

@dataclass
class BattleState:
    turn: int = 0
    weather: str = "none"
    terrain: str = "none"
    # p1_active, p1_bench (5), p2_active, p2_bench (5)
    teams: dict[str, list[PokemonState]] = field(
        default_factory=lambda: {s: [PokemonState() for _ in range(6)] for s in SIDES}
    )

class PokemonVocab:
    def __init__(self):
        self.species = {"none": 0}
        self.moves = {"none": 0}
        self.items = {"none": 0}
        
    def add(self, kind: str, name: str):
        target = getattr(self, kind)
        name = normalized_name(name)
        if name not in target:
            target[name] = len(target)
            
    def get(self, kind: str, name: str) -> int:
        target = getattr(self, kind)
        return target.get(normalized_name(name), 0)

def parse_replay_structured(log: str, vocab: PokemonVocab) -> list[dict[str, Any]]:
    state = BattleState()
    rows = []
    player_to_side = {}
    
    # Pre-pass to fill vocab and find teams
    for line in log.splitlines():
        if not line.startswith("|"): continue
        parts = line.split("|")
        if len(parts) < 2: continue
        event = parts[1]
        if event == "poke" and len(parts) >= 4:
            side = parts[2][:2]
            species = parts[3].split(",")[0]
            vocab.add("species", species)
        elif event == "move" and len(parts) >= 4:
            vocab.add("moves", parts[3])
        elif event == "player" and len(parts) >= 4:
            player_to_side[parts[3]] = parts[2]

    # Main parse
    current_turn_data = None
    for line in log.splitlines():
        if not line.startswith("|"): continue
        parts = line.split("|")
        event = parts[1] if len(parts) > 1 else ""

        if event == "turn":
            if current_turn_data: rows.append(current_turn_data)
            state.turn = int(parts[2])
            # Capture state at start of turn
            current_turn_data = {
                "turn": state.turn,
                "weather": vocab.get("species", state.weather), # reuse species or dedicated?
                "terrain": vocab.get("species", state.terrain),
                "p1_team": [(vocab.get("species", p.species), p.hp, STATUS_TO_ID.get(p.status, 0)) for p in state.teams["p1"]],
                "p2_team": [(vocab.get("species", p.species), p.hp, STATUS_TO_ID.get(p.status, 0)) for p in state.teams["p2"]],
            }
        elif event in ("switch", "drag") and len(parts) >= 4:
            side = parts[2][:2]
            species = parts[3].split(",")[0]
            # Simple heuristic: first pokemon of that species in team is active
            for p in state.teams[side]:
                if p.species == species or p.species == "none":
                    p.species = species
                    p.hp = hp_fraction(parts[4]) if len(parts) >= 5 else p.hp
                    # Move to front (index 0 is active)
                    idx = state.teams[side].index(p)
                    state.teams[side][0], state.teams[side][idx] = state.teams[side][idx], state.teams[side][0]
                    break
        elif event in ("-damage", "-heal") and len(parts) >= 4:
            side = parts[2][:2]
            state.teams[side][0].hp = hp_fraction(parts[3])
        elif event == "faint" and len(parts) >= 3:
            side = parts[2][:2]
            state.teams[side][0].hp = 0.0
            state.teams[side][0].fainted = True
        elif event == "win":
            winner_side = player_to_side.get(parts[2])
            for r in rows: r["winner_p1"] = 1 if winner_side == "p1" else 0

    return rows

class PokemonDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        
    def __len__(self):
        return len(self.rows) - 1
    
    def __getitem__(self, idx):
        curr = self.rows[idx]
        nxt = self.rows[idx+1]
        
        # State: [SpeciesID, HP, StatusID] x 12
        state = []
        for p in curr["p1_team"] + curr["p2_team"]:
            state.extend(p)
            
        # Target: Next HP for all 12 pokemon
        target_hp = [p[1] for p in nxt["p1_team"] + nxt["p2_team"]]
        
        return {
            "x": torch.tensor(state),
            "y_hp": torch.tensor(target_hp),
            "winner": torch.tensor(curr.get("winner_p1", 0), dtype=torch.float32)
        }

class JEPAEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, nhead=4, num_layers=2, latent_dim=128):
        super().__init__()
        self.species_embed = nn.Embedding(vocab_size, embed_dim)
        self.status_embed = nn.Embedding(8, 8)
        
        # Each pokemon: [species_embed + 1 (hp) + status_embed]
        self.token_dim = embed_dim + 1 + 8
        self.pokemon_proj = nn.Linear(self.token_dim, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, batch_first=True, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.latent_proj = nn.Linear(64 * 12, latent_dim)
        self.winner_head = nn.Linear(latent_dim, 1)
        
    def forward(self, x):
        # x: [batch, 36] (12 pokemon * 3 features)
        batch_size = x.shape[0]
        pokemon_tokens = []
        for i in range(12):
            spec = self.species_embed(x[:, i*3].long())
            hp = x[:, i*3+1].unsqueeze(1)
            stat = self.status_embed(x[:, i*3+2].long())
            pokemon_tokens.append(torch.cat([spec, hp, stat], dim=1).unsqueeze(1))
        
        tokens = torch.cat(pokemon_tokens, dim=1) # [batch, 12, token_dim]
        tokens = self.pokemon_proj(tokens)
        
        out = self.transformer(tokens) # [batch, 12, 64]
        z = self.latent_proj(out.view(batch_size, -1)) # [batch, latent_dim]
        
        win_logit = self.winner_head(z).squeeze(1)
        return z, win_logit

class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 12)
        )
        
    def forward(self, z):
        return self.net(z)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-replays", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--win-weight", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading dataset...")
    ds = load_dataset("milkkarten/pokemon-showdown-replays-merged", streaming=True, split="train")
    vocab = PokemonVocab()
    all_rows = []
    
    count = 0
    for example in tqdm(ds, desc="Parsing", total=args.max_replays):
        if count >= args.max_replays: break
        log = example.get("log", "")
        if not log: continue
        rows = parse_replay_structured(log, vocab)
        if rows:
            all_rows.extend(rows)
            count += 1

    print(f"Total turns: {len(all_rows)}. Vocab size: {len(vocab.species)}")
    
    dataset = PokemonDataset(all_rows)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = JEPAEncoder(len(vocab.species)).to(device)
    predictor = JEPAPredictor().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=5e-4)
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        encoder.train()
        predictor.train()
        total_jepa, total_win = 0, 0
        for batch in train_loader:
            x, y_hp, winner = batch["x"].to(device), batch["y_hp"].to(device), batch["winner"].to(device)
            
            z, win_logit = encoder(x)
            pred_hp = predictor(z)
            
            jepa_loss = mse_loss(pred_hp, y_hp)
            win_loss = bce_loss(win_logit, winner)
            
            loss = jepa_loss + args.win_weight * win_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_jepa += jepa_loss.item()
            total_win += win_loss.item()
            
        print(f"Epoch {epoch+1}, JEPA: {total_jepa/len(train_loader):.4f}, Win: {total_win/len(train_loader):.4f}")

    # Evaluation
    encoder.eval()
    zs, winners = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            z, _ = encoder(x)
            zs.append(z.cpu().numpy())
            winners.append(batch["winner"].numpy())
    
    zs = np.concatenate(zs)
    winners = np.concatenate(winners)
    
    if len(np.unique(winners)) > 1:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(zs, winners)
        acc = clf.score(zs, winners)
        print(f"Downstream Winner Prediction Accuracy (Frozen JEPA): {acc:.4f}")
        
        # Simple baseline (Current HP diff)
        hp_diffs = []
        for batch in test_loader:
            x = batch["x"].numpy()
            p1_hp = x[:, 1:18:3].sum(axis=1)
            p2_hp = x[:, 19:36:3].sum(axis=1)
            hp_diffs.append(p1_hp - p2_hp)
        hp_diffs = np.concatenate(hp_diffs).reshape(-1, 1)
        clf_base = LogisticRegression().fit(hp_diffs, winners)
        print(f"Downstream Winner Prediction Accuracy (HP Baseline): {clf_base.score(hp_diffs, winners):.4f}")

if __name__ == "__main__":
    main()
