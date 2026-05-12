# Pokemon JEPA World Models

Small-scale JEPA-style world-model experiments on public Pokemon Showdown replay
logs.

The project asks whether a compact Transformer learns a more useful battle-state
representation when it predicts next-turn state movement, rather than only the
eventual winner. The strongest result is a frozen-latent dynamics probe: across
replay budgets, the JEPA latent preserves substantially more next-turn HP
information than a winner-only supervised latent.

## What Is Included

- `code/run_pokemon_jepa_v3_study.py`: main replay-budget study comparing JEPA
  and supervised encoders.
- `code/generate_visual_maps.py`: latent-space PCA map and HP-delta map
  generation.
- `code/generate_svg.py`: dependency-free SVG diagrams for the blog post.
- `results/`: lightweight result JSON and plots used in the write-up.
- `figures/`: generated schematic SVGs.
- `summary.md`, `results.md`, `story_examples.md`: experiment notes and example
  analysis.

Cached replay data and model checkpoints are intentionally excluded.

## Data

The scripts stream or cache data from:

- Hugging Face dataset:
  `milkkarten/pokemon-showdown-replays-merged`
- Format filter used in the main study: `[Gen 9] PU`
- Parsed sample in the blog run: 991 replays, 20,831 turn transitions after
  filtering

The default cache path is `data/pokemon_jepa/`, which is ignored by git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch CUDA wheels depend on your local CUDA setup. Install the correct Torch
build for your machine if the default `pip install torch` is not appropriate.

## Reproduce The Main Study

```bash
python code/run_pokemon_jepa_v3_study.py \
  --max-replays 1000 \
  --budgets 25,50,100,250,500 \
  --epochs 8 \
  --refresh
```

This writes updated plots and JSON into `results/`.

To regenerate the latent-space and HP-delta maps:

```bash
python code/generate_visual_maps.py \
  --max-replays 1000 \
  --train-replays 500 \
  --epochs 8
```

To regenerate the schematic SVGs:

```bash
python code/generate_svg.py
```

## Headline Result

The main blog-facing result is not winner accuracy. Under a replay-held-out
split, winner accuracy is noisy. The useful signal is that the JEPA objective
changes what the latent preserves: a ridge probe from frozen JEPA latents to
next-turn HP has lower MSE than the same probe from winner-only supervised
latents across the tested replay budgets.
