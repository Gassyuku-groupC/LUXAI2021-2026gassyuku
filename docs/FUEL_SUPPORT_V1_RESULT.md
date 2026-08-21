# Fuel Support v1 Result

## Goal

Create a safer non-intrusive improvement layer from the current strongest agent:

`outputs/auto_league_dagger_v10_shadow/best_agent`

Unlike risk gate v1/v2, this experiment does not block `bw` and does not block `bcity`.

## Experiment Agent

Path:

`outputs/fuel_support_v1_from_best_agent`

Rule:

- Keep all worker and city production actions unchanged.
- Near night or during night, if a cargo-carrying worker is already standing on a friendly city tile whose city has low fuel buffer, prevent that worker from moving away.
- The intended effect is to preserve emergency fuel deposit without harming worker production.

Current config:

- `fuel_support_enabled: True`
- `fuel_support_mode: block`
- `fuel_support_max_turn: 200`
- `fuel_support_city_fuel_turns_lt: 5.0`
- `fuel_support_min_cargo_fuel: 20.0`
- `fuel_support_turns_to_night_lte: 3`
- `fuel_support_include_night: True`

## Smoke Test

Seed:

`1259068876`

Map/opponents:

- `16x16`
- `first`
- `stage400`
- `v4_stage350`
- both sides

Result:

- vs first: win_rate 1.0, effective_survival_rate 1.0
- vs stage400: win_rate 0.5, effective_survival_rate 1.0
- vs v4_stage350: win_rate 1.0, effective_survival_rate 1.0
- shadow_safe: true

This matches the best-agent baseline on the same seed.

## Interpretation

Fuel support v1 did not cause the catastrophic regression seen in direct `bw` blocking.

It is safer because it does not reduce worker production or expansion directly. However, on the smoke seed it did not visibly improve the aggregate metrics, likely because the rule only triggers in narrow states.

## Next Step

Use this as the base for a slightly stronger support layer:

- Log every would-hold worker event.
- Add a second dry-run rule for adjacent cargo workers that can move onto a low-fuel city tile.
- Keep any movement override in dry-run until it validates across random seeds.
