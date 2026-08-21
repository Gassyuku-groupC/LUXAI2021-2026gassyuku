# Project Route And Cleanup

## Current Production Agent

Keep:

- `outputs/auto_league_dagger_v10_shadow/best_agent`

Do not overwrite this directory. It is the current strongest safe agent and the baseline for all future diagnostic and gate experiments.

Optional local benchmark agents to keep for older evaluation scripts:

- `outputs/auto_league_dagger_v7_16x16/best_agent`
- `outputs/auto_league_dagger_v4_16x16/learner_agent`

These should be kept without their intermediate `game_stage_*` training folders.

## Active Direction

The active direction is no longer direct BC/Actor fine-tuning. The current route is:

1. Build strategy labels from replay data.
2. Train independent tabular scorers.
3. Score best-agent replay states and candidate action classes.
4. Validate high-risk windows with dry-run gate logs.
5. Only after fresh-seed validation, consider a very narrow runtime gate.

Core scripts:

- `scripts/build_strategy_label_dataset.py`
- `scripts/validate_strategy_labels.py`
- `scripts/train_strategy_label_scorers.py`
- `scripts/score_strategy_label_scorers.py`
- `scripts/train_strategy_candidate_scorers_v2.py`
- `scripts/score_candidate_actions_for_best.py`
- `scripts/validate_candidate_action_suggestions.py`
- `scripts/evaluate_generalization_promotion.py`
- `scripts/dry_run_gate_v1.py`

Core generated local artifacts:

- `dataset/processed/strategy_label_dataset_v1.csv`
- `outputs/diagnostic_layer/strategy_label_scorers_v1_16`
- `outputs/diagnostic_layer/strategy_candidate_scorers_v2_16`
- `outputs/diagnostic_layer/best_agent_candidate_action_suggestions_v2_16`
- `outputs/diagnostic_layer/best_agent_dry_run_gate_v1_16`

These artifacts are local and are ignored by Git. Recreate them from scripts when needed.

## Retired Directions

These directions were useful experimentally but should not drive the next work:

- direct BC / actor-only BC from best
- counterfactual BC v1-v4
- expansion suggestion BC v1
- residual head v1/v2 smoke branches
- reward-only RL tuning v13-v19
- hard action gates and direct movement overrides
- old risk-rule branches v1-v7

The main reason is consistent: they improved one diagnostic slice while damaging the learned base policy, reducing city scale, win rate, or side stability.

## Cleanup Policy

Large generated data should not be committed:

- `dataset/`
- `outputs/`
- `replays/`
- root `*.pt`
- logs and cache directories

Before a GitHub sync, use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\cleanup_project_for_route.ps1
```

This previews what would be deleted. To actually clean:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\cleanup_project_for_route.ps1 -Execute
```
