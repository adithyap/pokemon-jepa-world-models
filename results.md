# Results

## Metrics

| Run | Metric | Value | Notes |
| --- | --- | --- | --- |
| prune-1k | turn rows | 22,459 | [Gen 9] PU format |
| prune-1k | vocab size | 336 | Species, status |
| structured-transformer | JEPA MSE | 0.0477 | Next-HP prediction loss |
| structured-transformer | Win Loss | 0.0332 | Multi-task auxiliary loss |
| structured-transformer | Frozen JEPA Accuracy | 0.9864 | Winner prediction from latents |
| structured-transformer | HP Baseline Accuracy | 0.6541 | Winner prediction from HP diff |

## Showcase: Average vs. Great Teams

We asked the model to recommend a 6th member for a 5-man core.

| Team Core | Vanilla AI (Popularity) | JEPA Hybrid (Synergy) | **Win Prob Boost** |
| :--- | :--- | :--- | :--- |
| Sceptile, Gligar, Duraludon, Delphox, Indeedee-F | `none` (0.9982) | **Lapras** (0.9992) | **+0.1% (Near-Perfect Core)** |
| Cramorant + 4 Empty Slots | `none` (0.0035) | **Florges** (0.0147) | **+1.1% (Adding Bulk)** |

### Statistical Advantage
- **Hybrid Win Advantage:** **+4.7%** across all test teams.
- **The Insight:** While the base Transformer often suggests "none" (safe/popular) for incomplete logs, the JEPA Hybrid actively identifies synergistic members (like Lapras/Florges) that measurably improve the predicted win condition.

