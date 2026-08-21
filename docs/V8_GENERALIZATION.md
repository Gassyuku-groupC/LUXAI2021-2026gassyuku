# V8 Generalization Training

V8 treats `outputs/auto_league_dagger_v7_16x16/best_agent` as an immutable
16x16 stage 400 benchmark. Training and promotion outputs are written under
`outputs/auto_league_dagger_v8_generalization`; the v7 directory is never a
copy destination. The league verifies the stage 400 weights SHA-256 hash after
every stage.

## Training changes

- Training uses a seed-shuffled balanced cycle over 12x12, 16x16, 24x24,
  and 32x32. Every four completed environment episodes cover all four sizes.
- Each learner batch receives a random board reflection/rotation.
- Player 0 and player 1 axes are swapped with probability 0.5.
- On 12x12 and 16x16 maps, positive research progress below 200 receives an
  additional smooth reward. The existing 50 and 200 milestone rewards remain.
- V6/V7 loss normalization, clipped raw advantages, BC/RL balance, rollback,
  and the v4 fuel/survival reward remain active.
- Candidate agents disable inference-time `Rot180` ensembling. Training-time
  symmetry supplies invariance without doubling every large-map forward pass.

## Promotion gate

Every candidate plays both sides for every seed on the 12x12 and 16x16
promotion maps against:

- immutable v7 stage 400;
- v4 stage 350;
- the first-place agent.

Checks are applied separately to every map/opponent group. A high aggregate
score cannot hide a weak map, opponent, or side. The gate checks win rate,
effective survival, worst single-night city loss, small-map uranium completion,
and the normalized player 0/player 1 city-tile gap. A rejected candidate does
not replace either the best agent or the learner.

An early win counts as effective survival even when the replay ends before turn
360. `survived_360` remains available as the stricter replay-length metric.

## Run from VS Code

Open a PowerShell terminal in the repository and use a process-scoped execution
policy bypass:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\auto_train_generalization_v8.ps1 `
  -StartAtGames 15 `
  -TotalGames 111 `
  -GamesPerStage 16 `
  -Seeds 12345,23456,34567,45678
```

The default per-stage promotion evaluation covers 12x12 and 16x16. Training
samples all four legal map sizes. Keeping 24x24/32x32 out of every 16-game
promotion prevents one slow large-map replay from blocking the league.

To validate the training path quickly before a full run:

```powershell
.\scripts\auto_train_generalization_v8.ps1 `
  -StartAtGames 15 `
  -TotalGames 31 `
  -GamesPerStage 16 `
  -Seeds 12345,23456 `
  -EvaluationMapSizes 12,16
```

Run 24x24 and 32x32 periodically and explicitly, one seed and one map at a
time because the legacy replay runner can take longer than 900 seconds:

```powershell
.\scripts\auto_train_generalization_v8.ps1 `
  -TotalGames 15 `
  -GamesPerStage 15 `
  -EvalOnly `
  -Seeds 12345 `
  -EvaluationMapSizes 24 `
  -ReplayTimeoutSeconds 1800
```

Repeat with `-EvaluationMapSizes 32` after the 24x24 audit.

Large-map matches use a two-phase replay pipeline. The live match writes a
small command replay with `statefulReplay=false`; after the agents finish, the
local converter replays those commands without agent inference and writes the
final stateful JSON used by the evaluator and official visualizer. The
converter explicitly preserves fixed `width` and `height`, unlike the original
2021 converter.

Validated locally with stage 400 versus the public bot:

- 24x24: about 19 seconds for the match and 1 second for conversion;
- 32x32: about 15 seconds for the match and 1 second for conversion.

Use both large maps in one audit after the one-map checks pass:

```powershell
.\scripts\auto_train_generalization_v8.ps1 `
  -TotalGames 15 `
  -GamesPerStage 15 `
  -EvalOnly `
  -Seeds 12345 `
  -EvaluationMapSizes 24,32
```

This large-map result is an audit and should not replace the completed 12/16
promotion report. After large-map runtime is stable, use
`audit_stage400_generalization.ps1` for the combined 24x24/32x32 audit.
