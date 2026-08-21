# Fuel Support v3 Candidate Rules

## Goal

Convert the strongest offline miner signal into a candidate movement-support rule without changing agent actions.

Experiment package:

`outputs/fuel_support_v3_from_best_agent`

v3 is configured as dry-run only:

- `fuel_support_mode: dry_run`
- no runtime movement override
- no `bw` blocking
- no `bcity` blocking

## Input Events

The candidate rules were built from adjacent fuel-support miner outputs:

- `outputs/diagnostic_layer/fuel_support_v2_seed1259068876_16_fresh/adjacent_fuel_support_events.csv`
- `outputs/fuel_support_v1_from_best_agent/evaluation_random3_16/adjacent_fuel_support_events.csv`

Total input rows:

`593`

## Base Candidate Rule

Rule:

- event_type is `missed_adjacent_fuel`
- `city_fuel_turns < 2.5`
- `unit_cargo_fuel >= 80`
- `turn <= 160`

Output:

`outputs/fuel_support_v3_from_best_agent/candidate_movement_rules_v3`

Validation:

- rows: 293
- future_loss_10_rate: 0.7884
- future_big_loss_10_rate: 0.2355
- mean_future_loss_10: 3.3857
- pre-night rows: 105
- pre-night future_loss_10_rate: 0.8286

## Safer Variant: No-Action Workers

Rule:

- base candidate rule
- plus worker had no action in the command stream

Output:

`outputs/fuel_support_v3_from_best_agent/candidate_movement_rules_v3_no_action`

Validation:

- rows: 178
- future_loss_10_rate: 0.8708
- future_big_loss_10_rate: 0.3146
- mean_future_loss_10: 4.2079

This is the best first runtime candidate because it does not redirect an already chosen movement. It only proposes support for an adjacent worker that was otherwise idle/no-op.

## Other Variants

Pre-night only:

`outputs/fuel_support_v3_from_best_agent/candidate_movement_rules_v3_pre_night`

- rows: 105
- future_loss_10_rate: 0.8286
- mean_future_loss_10: 3.6286

Strict fuel/cargo:

`outputs/fuel_support_v3_from_best_agent/candidate_movement_rules_v3_strict`

- `city_fuel_turns < 1.5`
- `unit_cargo_fuel >= 100`
- rows: 69
- future_loss_10_rate: 0.7536
- mean_future_loss_10: 2.7536

The strict version is smaller but not stronger, so it is not the best first candidate.

## Candidate Spec

Primary candidate spec:

`outputs/fuel_support_v3_from_best_agent/candidate_movement_rules_v3_no_action/movement_override_candidate_spec.json`

Recommended next dry-run rule:

- name: `adjacent_low_fuel_cargo_support_v3`
- mode: `dry_run`
- action: `move_worker_to_adjacent_low_fuel_city_tile`
- `city_fuel_turns < 2.5`
- `unit_cargo_fuel >= 80`
- `turn <= 160`
- require original worker action to be empty/no-op

## Conclusion

Fuel support v3 should not yet execute movement overrides in battle.

The no-action-worker variant is a strong candidate signal:

- high loss correlation
- avoids touching `bw`, `bcity`, transfer, or already selected moves
- likely lower collision risk than redirecting active movement

Next step:

Implement runtime dry-run event counting for the no-action rule, or implement a controlled v4 override that only moves an idle adjacent worker into an empty friendly low-fuel city tile after collision checks.
