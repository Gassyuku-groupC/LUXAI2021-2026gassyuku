# Gate-Centered RL And Learned Gate v1

## Goal

This branch starts two follow-up directions after the fixed-threshold runtime gate:

1. Continue RL from the current best/gate agent with a win-and-scale-centered reward.
2. Train a learned action-level gate scorer that predicts whether an action should be intercepted.

The key design constraint is to avoid another destructive actor fine-tuning loop. The base policy remains anchored to the current best agent through teacher KL/BC, while the runtime gate remains a narrow safety mechanism.

## V20 RL Objective

New reward class:

```text
WinScaleCatastrophicGuardReward
```

Main reward structure:

- final result is the dominant signal
- final city margin and unit margin are secondary signals
- city/unit growth gives only low-amplitude dense shaping
- city loss is penalized mainly when it is catastrophic
- losing after a catastrophic loss receives an extra penalty
- scale shortfall is soft, not a hard threshold
- scale bonus is granted only when min/p25 fuel buffer is acceptable
- large low-fuel late-game cities receive a narrow guard penalty

Training config:

```text
conf/conv_teacher_bc_dagger_v20_win_scale_gate.yaml
```

Launcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\auto_train_win_scale_gate_v20_from_best.ps1 `
  -TotalGames 32 `
  -GamesPerStage 16 `
  -EvaluationEveryGames 16 `
  -RandomSeedCount 4 `
  -EvaluationMapSizes 12,16,24 `
  -ReplayTimeoutSeconds 300 `
  -ContinueOnReplayFailure
```

Promotion should prioritize win rate and final city/unit scale. Night loss remains a guardrail, not the sole objective.

## Learned Action Gate Scorer

New script:

```text
scripts/train_action_gate_scorer.py
```

The label is `action_gate_intervene`, not simply future city loss.

Positive examples:

- candidate action is gateable, currently `bw` or `bcity`
- future contains big city loss or failed big-loss label
- final outcome is weak or future city scale does not recover

Negative examples:

- same action class without future loss
- high-risk state that still wins or keeps scale
- safe expansion / stable-scale states
- non-gateable actions

Example all-map training command:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_action_gate_scorer.py `
  --input-csv .\dataset\processed\strategy_label_dataset_v1.csv `
  --output-dir .\outputs\diagnostic_layer\action_gate_scorer_v1_allmap `
  --map-size 0 `
  --gate-actions bw bcity `
  --threshold 0.30
```

Use the validation report to choose a conservative operating point. Early target:

- recall should be high enough to catch failed big-loss cases
- precision should not collapse, otherwise it becomes another over-conservative rule
- per-action alert rate should stay narrow, especially for `bcity`

Runtime integration should happen only after the scorer passes held-out replay validation and fresh-seed dry-run checks.
