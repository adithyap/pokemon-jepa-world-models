# Runs

## Environment

- GPU: NVIDIA GeForce GTX 1080 Ti 12GB
- Python: conda env `ml`, Python 3.12.13
- Torch: 2.11.0+cu126, CUDA available
- Commit: not recorded yet

## Commands

```bash
python code/run_pokemon_jepa.py --max-replays 50 --epochs 1 --format-filter gen9pu --refresh

python code/run_pokemon_jepa.py --max-replays 1000 --epochs 8 --format-filter gen9pu --refresh

python code/run_pokemon_jepa_v3_study.py --max-replays 1000 --budgets 25,50,100,250,500 --epochs 8

python code/generate_visual_maps.py --max-replays 1000 --train-replays 500 --epochs 8
```

## Run Log

| Run | Command/config | Dataset subset | Seed | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| init | initializer above | none | n/a | created/preserved scaffold | no experiment run yet |
| smoke | `run_pokemon_jepa.py --max-replays 50 --epochs 1 --format-filter gen9pu --refresh` | `milkkarten/pokemon-showdown-replays-merged`, `data/part-00000.parquet`, `[Gen 9] PU` | 7 | 50 replays, 1,092 rows, CUDA train completed | proved parser/training path; JEPA underperformed raw features |
| prune-1k | `run_pokemon_jepa.py --max-replays 1000 --epochs 8 --format-filter gen9pu --refresh` | same shard and format | 7 | 1,000 replays, 22,459 rows, CUDA train completed | metrics in `results.md`; result JSON at `results/pokemon_jepa_results.json` |
