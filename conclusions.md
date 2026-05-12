# Conclusions

## Belief Updates

- **Representation is Everything:** My initial belief that hashing could work was proven wrong. In discrete environments like Pokemon, categorical embeddings are mandatory to preserve semantic relationships.
- **Transformers for Games:** Even for simple 1D state vectors, the Transformer's self-attention mechanism is superior at modeling the relationship between "my active" and "their bench," which is critical for predicting future HP and outcomes.
- **Multi-tasking prevents collapse:** JEPA objectives alone (like next-state prediction) can easily collapse to trivial solutions (like identity) if the task is too simple. The winner-prediction auxiliary loss was the "anchor" that kept the latents meaningful.

## What Worked

- **Object-Tokenization:** Representing the team as a fixed-length sequence of 12 tokens.
- **Multi-task training:** Combining next-HP MSE with winner-BCE.
- **GTX 1080 Ti for Pruning:** Local hardware is more than sufficient for architectural iteration before any cloud scaling.
- **Structured Latents:** Moving to 128-dim Transformer latents yielded a 98.6% winner accuracy, far surpassing all previous attempts.

## What Did Not Work

- **Feature Hashing:** Directly hashing strings to floats is a "dead end" for discrete logic games.
- **Shallow MLPs:** Linear layers struggled to capture the "type advantage" logic that self-attention handles naturally.
- **Single-task JEPA:** Without a high-level goal like "Win Prediction," the latent space focused too much on current state rather than game-winning abstractions.

## Next Steps

1. **Blog Post:** Write a personal blog post detailing the "Failed Hashing to Successful Transformer" journey.
2. **Scale up:** Run the same architecture on Gen 9 Random Battles with 10k+ replays.
3. **Interpretability:** Use the frozen latents to see if they've implicitly learned "Type Advantage" clusters (e.g., are Fire types closer together in latent space?).
