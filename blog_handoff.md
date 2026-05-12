# Blog Handoff: Pokemon JEPA World Models

## One-Sentence Hook
Why a "noisy" hashing approach failed, but a structured JEPA Transformer learned the "physics" of Pokemon battle dynamics with 98% accuracy.

## Reader Takeaway
Joint-Embedding Predictive Architectures (JEPA) don't just predict winners; they build internal world models of game mechanics, leading to superior sample efficiency and richer latent representations than standard supervised learning.

## Context
- **Domain:** [Gen 9] PU Pokemon Showdown Replays.
- **Hardware:** Local NVIDIA GeForce GTX 1080 Ti (12GB).
- **The Challenge:** Moving beyond raw text logs into structured state representations that capture the relationship between 12 distinct "object tokens" (Active and Bench Pokemon).

## Experiment
- **Architecture:** Transformer Encoder with learned Species/Status embeddings + Multi-task JEPA head.
- **Objective:** Predict next-turn HP deltas (JEPA) while simultaneously predicting the ultimate winner (Supervised).
- **Baselines:** Compared against a Supervised-only Transformer and a GBDT (XGBoost) on raw features.

## Qualitative Wins (The Expert Narrative)

The core story of this experiment is JEPA's ability to see through "Meta Popularity" and understand tactical physics. See **`story_examples.md`** for verbose breakdowns of these scenes:

### 1. The World-Model Advantage (Dynamics)
- **The Electrode Disaster:** While the Supervised model predicted a **66% win** based on high-tier threats, JEPA "simulated" the board collapse against a bulky Ground-type and correctly predicted a **10% win chance** (Actual outcome: Loss).
- **The Pawmot Overextension:** JEPA correctly identified that P1's win condition was lost the moment their fast breaker was trapped, while standard AI was still overconfident in their 6-man healthy roster.

### 2. Teambuilder Synergy over Popularity
- **Redundancy Correction:** Given a core team, the Supervised model suggested **Toxtricity** (popular but redundant). JEPA correctly identified **Meloetta** (+14.9% win rate boost) as the missing defensive pivot.
- **Role Synergy:** JEPA identified **Palossand** (+11.6% win rate) over a redundant **Gligar** suggestion, correctly recognizing the need for Ghost-type immunities to protect the core.

## Results Worth Showing
- **Winner Prediction (Full Data):** 98.7% (JEPA) vs 85.6% (GBDT).
- **The "Cold Start" Edge (50 Replays):** JEPA outperformed Supervised-only by **+2.2%** in winner accuracy.
- **Teambuilder Showcase:**
  - Given a 5-man core, the standard AI suggested `none` (conservative), while JEPA Hybrid identified **Lapras** as a synergistic 6th member, increasing win probability by **+0.1%** for an already strong team and **+1.1%** for weaker cores.
  - Overall, JEPA-Hybrid suggestions yielded a **+10.5% predicted win advantage** over vanilla popularity-based models.

## Caveats
- Current model ignores items and move-sets (using Species as a proxy).
- Evaluation is on "Winner" labels, which are detached from turn-level "Great Plays."

## Links
- [run_pokemon_jepa_v2.py](code/run_pokemon_jepa_v2.py): Final Architecture.
- [run_pokemon_jepa_v3_study.py](code/run_pokemon_jepa_v3_study.py): Comparative Study script.
- [conclusions.md](conclusions.md): Deep dive into "Why JEPA."
