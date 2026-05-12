# Candidate Ideas

| Idea | Sources | Dataset | Local feasibility | Hypothesis | Metrics | Score |
| --- | --- | --- | --- | --- | --- | --- |
| Pokemon object-masked JEPA | Causal-JEPA; TD-JEPA | `milkkarten/pokemon-showdown-replays-merged` streaming Gen 9 random battles; fallback Kaggle Gen9 Randbats | High: stream 10k-50k battles, extract compact turn objects, train small Transformer/MLP latent predictor | Masking future Pokemon/team/field objects forces latent interaction structure that improves winner and next-action probes over text n-gram or one-step state baselines. | winner AUC/log loss, next action top-k, HP/status/field reconstruction F1, probe accuracy vs frozen baseline | 28/30 |
| Chess eval-delta JEPA | TD-JEPA; Lichess evals | `cetusian/chess-sft-lichess-2200` plus sampled Lichess eval JSONL positions | High: 200k rows fit locally; optional stream evals | A latent future-state predictor trained on FEN transitions captures tactical value changes better than supervised next-move-only embeddings. | centipawn sign accuracy, eval bucket MAE, legal move top-k, linear probe Elo-bin accuracy | 26/30 |
| Chess piece-object Causal-JEPA | Causal-JEPA | Lichess PGN or FEN-derived board tensors | Medium: parsing and legal-transition generation are straightforward but object masking design needs care | Masking individual piece objects and predicting them from other pieces should improve tactical motif probes versus square-patch masking. | piece reconstruction, attacked-square F1, puzzle theme probe accuracy, eval sign accuracy | 25/30 |
| Pokemon TD-JEPA for long-horizon win probability | TD-JEPA | `milkkarten/pokemon-showdown-replays-merged` Gen 9 OU or random battles | Medium-high: offline transitions, no simulator required | TD-style latent targets over several future turns will learn more stable win-probability representations than one-step next-event prediction. | win-prob log loss by turn, calibration ECE, temporal consistency, late-game generalization | 25/30 |
| Verifiable Pokemon world model consistency | RLVR-World; Pokemon Showdown parser | Kaggle parsed Gen9 Randbats or streamed Showdown logs | Medium: reward/checker engineering is more work than training | Parser-verifiable constraints such as legal faint/switch/action consistency can improve latent rollout quality without full simulator integration. | invalid transition rate, next legal action accuracy, reconstruction loss, winner AUC | 23/30 |
| Chess RLVR legality-aligned latent predictor | RLVR-World; Lichess | Lichess positions, python-chess legal move generator | Medium: legality checks are easy, but RL-style optimization may be overkill locally | Verifiable legal-move and checkmate constraints can align latent dynamics beyond maximum-likelihood next move. | illegal move rate, legal top-k, mate-in-n probe, eval bucket accuracy | 22/30 |
| Cross-game JEPA transfer: chess to Pokemon abstractions | seq-JEPA; Causal-JEPA | Small Lichess subset plus Pokemon subset | Low-medium: interesting but likely diffuse | Object-role pretraining on chess transfers weakly to Pokemon object interaction probes if the architecture captures game-state relational structure. | transfer probe lift, sample efficiency at 1k/5k battles, representation similarity | 20/30 |
| Pokemon replay language-token JEPA baseline | seq-JEPA | Raw `log` field from Pokemon Showdown datasets | High technically, lower scientific value | Latent prediction over replay text may perform well on next-event metrics but learn brittle surface syntax instead of world state. | next log event perplexity/proxy loss, winner AUC, parser-valid rollout rate | 20/30 |

## Score Breakdown

| Idea | Personal fit | Novelty | Dataset | Local feasibility | Blog potential | Scale-up value | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pokemon object-masked JEPA | 5 | 5 | 5 | 4 | 5 | 4 | 28 |
| Chess eval-delta JEPA | 5 | 4 | 5 | 5 | 4 | 3 | 26 |
| Chess piece-object Causal-JEPA | 5 | 5 | 5 | 4 | 4 | 2 | 25 |
| Pokemon TD-JEPA for long-horizon win probability | 5 | 4 | 5 | 4 | 4 | 3 | 25 |
| Verifiable Pokemon world model consistency | 5 | 4 | 4 | 3 | 4 | 3 | 23 |
| Chess RLVR legality-aligned latent predictor | 4 | 4 | 5 | 4 | 3 | 2 | 22 |
| Cross-game JEPA transfer | 5 | 4 | 4 | 2 | 4 | 1 | 20 |
| Pokemon replay language-token JEPA baseline | 4 | 3 | 5 | 5 | 2 | 1 | 20 |

## Selection

1. Local prune first: Pokemon object-masked JEPA.
2. Backup if Pokemon parsing slows down: Chess eval-delta JEPA.
3. Keep in reserve: Pokemon TD-JEPA for long-horizon win probability.
