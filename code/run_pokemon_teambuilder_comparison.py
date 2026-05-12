#!/usr/bin/env python3
"""Teambuilder Comparison: JEPA vs Standard Baselines.

Compares Top-K accuracy for predicting the 6th Pokemon:
1. JEPA-Full (JEPA-HP + Winner-Label + MLM)
2. Winning-MLM (Winner-Label + MLM only)
3. Vanilla-MLM (MLM only - Standard BERT style)
"""

from __future__ import annotations

import argparse
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

# Importing components from our previous teambuilder script logic
from run_pokemon_jepa_teambuilder import (
    PokemonVocab, parse_replay_teambuilder, TeambuilderDataset, TeambuilderModel
)

def train_variant(name, model, loader, epochs, device, use_jepa=True, use_win=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    
    print(f"Training {name} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_l = 0
        pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            x, y_hp, win = batch["x"].to(device, non_blocking=True), batch["y_hp"].to(device, non_blocking=True), batch["winner"].to(device, non_blocking=True)
            
            x_masked = x.clone()
            mask_targets = []
            for i in range(x.shape[0]):
                idx = random.randint(0, 5)
                mask_targets.append(x[i, idx*8].item())
                x_masked[i, idx*8] = 1 
            mask_targets = torch.tensor(mask_targets, device=device).long()

            out = model(x_masked)
            loss = ce(out["mlm"], mask_targets)
            if use_jepa: loss += mse(out["hp"], y_hp)
            if use_win:  loss += 0.5 * bce(out["win"], win)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_l += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"  {name} Epoch {epoch+1}/{epochs} | Avg Loss: {total_l/len(loader):.4f}", flush=True)

def evaluate_mlm(model, loader, device, k=10):
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating MLM", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            for i in range(x.shape[0]):
                target = x[i, 5*8].item()
                if target <= 1: continue 
                x_test = x[i:i+1].clone(); x_test[0, 5*8] = 1 
                out = model(x_test)
                top_k = torch.topk(out["mlm"], k).indices[0]
                if target in top_k.tolist(): hits += 1
                total += 1
    return hits / total if total > 0 else 0

def evaluate_hybrid(model_j, model_v, loader, device):
    model_j.eval(); model_v.eval()
    j_win_probs, v_win_probs, h_win_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating Hybrid Synergy", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            for i in range(min(20, x.shape[0])): 
                x_test = x[i:i+1].clone()
                x_test[0, 5*8] = 1 
                
                out_j = model_j(x_test)
                out_v = model_v(x_test)
                
                s_j = torch.topk(out_j["mlm"], 1).indices[0] 
                s_v = torch.topk(out_v["mlm"], 1).indices[0] 
                
                s_top10 = torch.topk(out_j["mlm"], 10).indices[0]
                batch_x = x_test.repeat(len(s_top10), 1)
                for j, s_id in enumerate(s_top10): batch_x[j, 5*8] = s_id
                
                win_scores = torch.sigmoid(model_j(batch_x)["win"])
                s_hybrid = s_top10[torch.argmax(win_scores)].unsqueeze(0)
                
                def score(model, base_x, s_id):
                    tx = base_x.clone(); tx[0, 5*8] = s_id
                    return torch.sigmoid(model(tx)["win"]).item()

                j_win_probs.append(score(model_j, x_test, s_j))
                v_win_probs.append(score(model_j, x_test, s_v))
                h_win_probs.append(score(model_j, x_test, s_hybrid))
            break
                
    return np.mean(j_win_probs), np.mean(v_win_probs), np.mean(h_win_probs)

def run_regime(name, tr_ds, te_ld, vocab, device, epochs=10):
    v_spec, v_item, v_move = len(vocab.species), len(vocab.items), len(vocab.moves)
    tr_ld_opt = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    
    print(f"\n--- Regime: {name} ---")
    m_j = TeambuilderModel(v_spec, v_item, v_move).to(device)
    train_variant("JEPA-Full", m_j, tr_ld_opt, epochs, device, True, True)
    m_v = TeambuilderModel(v_spec, v_item, v_move).to(device)
    train_variant("Vanilla-MLM", m_v, tr_ld_opt, epochs, device, False, False)
    
    acc_j = evaluate_mlm(m_j, te_ld, device)
    acc_v = evaluate_mlm(m_v, te_ld, device)
    win_j, win_v, win_h = evaluate_hybrid(m_j, m_v, te_ld, device)
    
    print(f"Results for {name}:")
    print(f"  Popularity (Top-10 Acc):  JEPA {acc_j:.4f} vs Vanilla {acc_v:.4f}")
    print(f"  Win Prob (Top-1 Suggest): JEPA {win_j:.4f} vs Vanilla {win_v:.4f} vs HYBRID {win_h:.4f}")
    print(f"  Hybrid Win Advantage:     {((win_h - win_v)/win_v*100 if win_v > 0 else 0):.1f}%")
    return m_j, m_v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-replays", type=int, default=1000)
    args = parser.parse_args()
    
    print("Loading and Parsing Dataset...")
    ds = load_dataset("milkkarten/pokemon-showdown-replays-merged", streaming=True, split="train", trust_remote_code=True, token=False)
    vocab, all_rows = PokemonVocab(), []
    for ex in tqdm(ds, desc="Parsing", total=args.max_replays):
        if len(all_rows) // 20 >= args.max_replays: break
        rows = parse_replay_teambuilder(ex.get("log", ""), vocab)
        if rows: all_rows.extend(rows)

    dataset = TeambuilderDataset(all_rows)
    tr_sz = int(0.8 * len(dataset))
    tr_ds, te_ds = torch.utils.data.random_split(dataset, [tr_sz, len(dataset)-tr_sz])
    te_ld = DataLoader(te_ds, batch_size=64, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run Comparison
    model_j, model_v = run_regime("1000 Replays", tr_ds, te_ld, vocab, device, epochs=10)

    # Showcase Section
    print("\n--- Showcase: Average vs. Great Teams ---")
    model_j.eval(); model_v.eval()
    inv_spec = {v: k for k, v in vocab.species.items()}
    
    with torch.no_grad():
        test_batch = next(iter(te_ld))
        x_all = test_batch["x"].to(device)
        
        for i in range(min(3, x_all.shape[0])):
            x_test = x_all[i:i+1].clone()
            x_test[0, 5*8] = 1 # Mask 6th
            
            out_j = model_j(x_test)
            out_v = model_v(x_test)
            
            s_v = torch.topk(out_v["mlm"], 1).indices[0]
            
            s_top10 = torch.topk(out_j["mlm"], 10).indices[0]
            batch_x = x_test.repeat(len(s_top10), 1)
            for j, s_id in enumerate(s_top10): batch_x[j, 5*8] = s_id
            win_scores = torch.sigmoid(model_j(batch_x)["win"])
            s_h = s_top10[torch.argmax(win_scores)].unsqueeze(0)
            
            def get_prob(m_id):
                tx = x_test.clone(); tx[0, 5*8] = m_id
                return torch.sigmoid(model_j(tx)["win"]).item()
            
            team_5 = [inv_spec[x_test[0, j*8].item()] for j in range(5)]
            print(f"\nTeam {i+1} Core: {team_5}")
            print(f"  Vanilla Suggestion: {inv_spec[s_v.item()]:<15} | Predicted Win: {get_prob(s_v):.4f}")
            print(f"  JEPA Hybrid Suggest: {inv_spec[s_h.item()]:<15} | Predicted Win: {get_prob(s_h):.4f}")
            if s_v != s_h:
                print(f"  --> JEPA identified a synergistic member that increased win prob by {((get_prob(s_h)-get_prob(s_v))*100):.1f}%")

if __name__ == "__main__":
    main()
