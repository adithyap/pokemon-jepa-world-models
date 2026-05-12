# Hypotheses

## Selected Hypothesis

- Claim: object-masked latent prediction on Pokemon battle trajectories learns interaction-aware state representations that improve simple downstream probes for winner and next action compared with a bag-of-events baseline and a one-step autoencoder.
- Falsification criteria: no statistically meaningful lift on winner log loss/AUC or next-action top-k after controlling for turn number, revealed team count, and HP buckets; representations collapse to turn-count proxies; parser extraction takes more than one day before any model signal.
- Minimal experiment: stream 10k Gen 9 random battle replays, extract per-turn object records, train a small latent predictor for 1-3 future turns with object masking, freeze encoder, fit logistic/linear probes.
- Expected local runtime: 2-4 hours for extraction and baseline probes; 2-6 hours for a small PyTorch model on GTX 1080 Ti depending on sequence length.

## Local Result

- Result: falsified for the current hashed 17-feature representation and 8-epoch masked future objective.
- Evidence: raw features beat JEPA embeddings on winner AUC (`0.729` vs `0.703`), winner log loss (`0.607` vs `0.628`), next-action accuracy (`0.465` vs `0.445`), and next-action top-2 accuracy (`0.779` vs `0.769`).
- Interpretation: the parser and CUDA pipeline are viable, but the representation is too compressed and hash-based to justify scale-up.

## Backup Hypothesis

- Claim: a chess latent predictor trained on FEN transition/eval deltas encodes board value better than a next-move-only supervised representation.
- Falsification criteria: eval sign/bucket probes are no better than piece-count/material baselines or next-move SFT embeddings.
- Minimal experiment: use `cetusian/chess-sft-lichess-2200`, convert PGN prefixes to board tensors with `python-chess`, train JEPA-style future-board/eval-delta objective, probe on held-out eval buckets.
- Expected local runtime: 1-3 hours if using the 200k-row dataset and a compact encoder.
