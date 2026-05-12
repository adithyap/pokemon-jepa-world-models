# Failures

| Attempt | Failure mode | Evidence | Next possible fix |
| --- | --- | --- | --- |
| all-shards streaming | `load_dataset` startup stalled before writing rows | initial run against the full dataset repo produced empty local files and no progress | load one explicit parquet shard with `--data-file data/part-00000.parquet` |
| Gen 9 random battle on shard 0 | format mismatch | first 50k rows in shard 0 were `[Gen 9] PU`, yielding 0 `gen9randombattle` rows | use `--format-filter gen9pu` or inspect later shards for random battles |
| current JEPA embedding | underperformed raw features | winner AUC `0.703` vs raw `0.729`; next-action accuracy `0.445` vs raw `0.465` | replace hashed scalar features with object tokens, categorical embeddings, side-relative labels, and richer targets |
