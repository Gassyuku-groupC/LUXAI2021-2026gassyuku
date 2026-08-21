# Risk Gate v1/v2 Result

## Goal

Test a non-intrusive post-policy safety gate on the current strongest agent:

`outputs/auto_league_dagger_v10_shadow/best_agent`

The gate was applied only to copied experiment agents. The production best agent was not modified.

## Experiments

### v1

Path:

`outputs/risk_gate_v1_from_best_agent`

Rule tested:

- Before or at turn 160
- If a city tile wants to execute `bw`
- And that city has `fuel / upkeep < 5`
- Replace `bw` with research when possible

Result on seed `1259068876`, map size 16, opponents `first,stage400,v4_stage350`:

- win_rate: 0.0 for all opponent groups
- effective_survival_rate: 0.0 for all opponent groups
- shadow_safe: false

The rule was too destructive. It reduced early worker growth and caused full collapse.

### v2

Path:

`outputs/risk_gate_v2_from_best_agent`

Rule tested:

- Same as v1, but stricter fuel threshold: `fuel / upkeep < 3`

Result on the same seed/opponents:

- win_rate: 0.0 for all opponent groups
- effective_survival_rate: 0.0 for all opponent groups
- shadow_safe: false

Even the stricter worker gate was still destructive.

## Baseline Comparison

Baseline path:

`outputs/auto_league_dagger_v10_shadow/best_agent`

Same seed `1259068876`, map size 16, opponents `first,stage400,v4_stage350`:

- vs first: win_rate 1.0, effective_survival_rate 1.0
- vs stage400: win_rate 0.5, effective_survival_rate 1.0
- vs v4_stage350: win_rate 1.0, effective_survival_rate 1.0
- shadow_safe: true

Therefore the regression came from the action gate, not from the evaluation seed.

## Conclusion

Do not use direct `bw` blocking as a safety gate.

Although `bw_low_fuel_lt5_high_risk` correlated with future city loss in the offline logs, it is not causally safe to block. In these states, building workers may be part of the recovery mechanism that preserves mining tempo and future fuel income.

Both v1 and v2 are now set to `risk_gate_mode: dry_run` to prevent accidental use as battle agents.

## Next Direction

The next safe improvement should avoid reducing worker production directly.

Prefer:

- Runtime logging only for risky `bw`
- Fuel-transfer or movement-level support around risky cities
- Expansion recommendation analysis for low-risk, high-buffer windows
- More replay-level causal checks before any action replacement
