# RunPod A100 Validation

Status: not warranted

## Trigger Criteria

- Local Pokemon object-masked JEPA beats simple baselines by at least 3-5 percent relative on winner log loss or next-action top-k.
- Parser coverage exceeds 90 percent on the chosen format.
- Scaling question is concrete: more games, longer context, or larger latent model changes the conclusion.

Current local run does not meet the trigger criteria. The learned JEPA embedding underperformed raw parsed features on the 1,000-replay pruning run.

## Proposed A100 Run

- GPU: A100 40GB
- Expected time: 5-10 hours only after local signal
- Dataset scale: 500k-2M Pokemon transitions or a larger streamed Lichess eval subset
- Commands: to be filled after local code exists
- Stop criteria: stop if validation probe lift plateaus for two consecutive checkpoints or parser-invalid rollout rate remains high
