# Breaking The Reward-Weight Loop

The recent v16-v18 runs show the same failure pattern:

- Increasing survival pressure reduces large night losses, but it can collapse weak-side expansion.
- Re-adding expansion pressure restores scale, but it reintroduces unsafe growth and night losses.
- Promotion metrics then reject the candidate for either `night_loss` or `side_city_gap`, so tuning cycles back to the opposite pressure.

This should not be treated as a scalar reward problem anymore. The agent needs a separate decision layer for:

1. unsafe actions that should be blocked or heavily discouraged;
2. missed safe expansion opportunities that should be encouraged;
3. normal policy behavior that should stay anchored to the current best agent.

## New Rule

Do not launch a new survival/expansion reward sweep unless an intervention report first identifies reproducible action-level failure modes.

Use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_intervention_validation_pipeline.ps1
```

This creates:

- `outputs\diagnostic_layer\break_reward_loop_v1\intervention_candidates`
- `outputs\diagnostic_layer\break_reward_loop_v1\intervention_validation`
- `outputs\diagnostic_layer\break_reward_loop_v1\gate_dry_run`

The pipeline validates up to 500 rows per candidate type by default. Increase `-TopN`
when checking a larger replay batch, rather than trusting only the highest-priority
examples.

## Promotion To Training

Only move from diagnostics to training when both are true:

- safety gate rules reproduce on fresh seeds with high future-loss rate;
- expansion suggestions identify low-risk, fuel-healthy missed scale opportunities without being concentrated on one side only.

If either condition fails, collect better labels or improve the scorer. Do not compensate by increasing `city_loss`, `late_expansion`, `risk_adjusted_city_gain`, or `teacher_bc_cost`.

## Practical Direction

The next robust improvement should be one of:

- a conservative runtime gate that blocks only validated high-risk `BUILD_WORKER` or adjacent low-fuel `BUILD_CITY` actions;
- a supervised auxiliary decision head for safe expansion/survival mode, trained from counterfactual labels;
- data rebalancing around side/phase failure buckets before any RL continuation.

Reward weights can still be used for light shaping, but they should no longer be the main mechanism for teaching expand-versus-survive.
