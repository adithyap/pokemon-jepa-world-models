# Pokemon JEPA World Models

Reproduction code for a small JEPA-style world-model study on Pokemon Showdown
replays.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Main replay-budget study:

```bash
python code/run_pokemon_jepa_v3_study.py \
  --max-replays 1000 \
  --budgets 25,50,100,250,500 \
  --epochs 8 \
  --refresh
```

Latent-space and HP-delta plots:

```bash
python code/generate_visual_maps.py \
  --max-replays 1000 \
  --train-replays 500 \
  --epochs 8
```

SVG schematics:

```bash
python code/generate_svg.py
```

Outputs are written to `results/` and `figures/`. Cached replay data is written
to `data/`; all three directories are ignored by git.
