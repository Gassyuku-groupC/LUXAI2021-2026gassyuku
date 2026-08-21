# V9 Robustness Training

V9 starts a new league from the immutable v7 stage 400 weights. Rejected V8
candidates are not used as learners.

## Reward changes

- On 12x12 and 16x16, crossing turn 80 below 50 research gives one
  proportional shortfall penalty.
- On 12x12 and 16x16, crossing turn 200 below 200 research gives one
  proportional shortfall penalty.
- A newly created standalone city is checked against the fuel required to
  survive the next full night. Its own proportional shortfall is penalized
  once, with smoothly increased weight from turn 60 through turn 120.
- V8's persistent positive small-map research reward is disabled. Existing
  milestone, terminal, fuel-buffer, and survival logic remains unchanged.

## Loss balancing

V-trace importance ratios continue to use the raw joint action log-probability.
Only the policy-gradient log-probability is divided by the actual action count
for each time step and player. This prevents 24x24 and 32x32 games from
receiving larger policy gradients merely because they contain more units and
city tiles.

## Promotion

The promotion gate remains unchanged. In particular,
`max_side_city_gap=0.35`; 0.45 is only an intermediate diagnostic target and
does not promote a candidate.

## Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\auto_train_robustness_v9.ps1 `
  -TotalGames 96 `
  -GamesPerStage 16 `
  -Seeds 12345,23456,34567,45678
```

Training uses balanced 12x12, 16x16, 24x24, and 32x32 self-play. Per-stage
promotion evaluates 12x12 and 16x16 against stage 400, v4 stage 350, and 1st
on both sides. Use `-EvaluationMapSizes 12,16,24,32` for a full four-map gate;
the two-phase replay pipeline prevents large-map stateful replay timeouts.
