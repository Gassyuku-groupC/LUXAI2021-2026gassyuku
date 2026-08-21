# Fuel Support V4 Result

## Goal

Test one very narrow non-neural intervention on top of the current best agent:

- allow normal worker/city behavior;
- only move an idle adjacent cargo-carrying worker into a low-fuel friendly city tile;
- require collision checks before adding the move.

The intended rule was deliberately small: support low-fuel cities without blocking `bw`, `bcity`, or existing worker actions.

## Implemented Agents

### `outputs/fuel_support_v4_from_best_agent`

Implemented actual override:

- `fuel_support_v4_override_enabled: True`
- `fuel_support_v4_city_fuel_turns_lt: 2.5`
- `fuel_support_v4_min_cargo_fuel: 80.0`
- `fuel_support_v4_max_turn: 160`
- `fuel_support_v4_require_no_action: True`

Collision checks:

- worker must have no existing action;
- worker must be adjacent to the target friendly city tile;
- worker must not already be standing on a friendly city tile;
- target tile must not contain own/enemy unit;
- target tile must not already be claimed by another planned move.

After evaluation, this branch was reverted to `fuel_support_v4_mode: dry_run`.

### `outputs/fuel_support_v4b_from_best_agent`

Same intervention as v4, but restricted to pre-night only:

- `fuel_support_v4_include_night: False`

After evaluation, this branch was also reverted to `fuel_support_v4_mode: dry_run`.

## Evaluation

Fixed smoke seed:

- seed: `1259068876`
- map: `16x16`
- output roots:
  - `outputs/diagnostic_layer/fuel_support_v4_seed1259068876_16_fresh`
  - `outputs/diagnostic_layer/fuel_support_v4b_seed1259068876_16_fresh`

## V4 Result

V4 passed shadow survival but failed side stability:

- vs first: win rate `0.5`, survival `1.0`, mean city tiles `43`, worst night loss `8`
- vs first side gap failed badly:
  - p0: win, `79` city tiles
  - p1: loss, `7` city tiles
- vs stage400: win rate `1.0`, survival `1.0`, mean city tiles `62`
- vs v4_stage350: win rate `1.0`, survival `1.0`, mean city tiles `64.5`

Compared with the current best/v2 fixed-seed baseline, the p1 vs first game regressed from a strong stable game to near-collapse. This is unsafe.

## V4B Result

V4B was worse than v4:

- vs first: win rate `0.5`, survival `1.0`, mean city tiles `44.5`, side gap failed
- vs stage400: win rate `0.5`, survival `1.0`, mean city tiles `31`, worst night loss `12`, side gap failed
- vs v4_stage350: win rate `0.0`, survival `1.0`, mean city tiles `25.5`

Restricting the rule to pre-night did not solve the instability. It reduced some night intervention, but still damaged expansion and side balance.

## Conclusion

The actual movement override is still too invasive, even though it only touches idle adjacent cargo workers.

Likely reason:

- idle workers are not always disposable;
- moving into a city can break the learned local mining/expansion rhythm;
- the p1 side is especially sensitive to tiny trajectory changes;
- fuel support must be evaluated as a counterfactual or recommendation first, not directly injected as a move.

Current decision:

- do not use `fuel_support_v4` or `fuel_support_v4b` as battle agents;
- keep both branches in `dry_run`;
- keep `auto_league_dagger_v10_shadow/best_agent` as the safe production agent.

Recommended next step:

- continue with offline replay mining and candidate validation;
- build a diagnostic recommendation layer that reports missed support opportunities;
- only consider runtime action changes after repeated multi-seed evidence shows the intervention improves p1 and side-gap stability.
