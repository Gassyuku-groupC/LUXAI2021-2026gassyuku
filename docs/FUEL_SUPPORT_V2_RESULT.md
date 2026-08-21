# Fuel Support v2 Result

## Goal

Extend `fuel_support_v1_from_best_agent` with a safer observation layer for future movement support.

v2 keeps the v1 behavior:

- Do not block `bw`
- Do not block `bcity`
- Only hold cargo workers that are already standing on critically low-fuel friendly cities near night

New in v2:

- Dry-run log for adjacent cargo workers that could move onto a low-fuel friendly city tile.
- The adjacent support rule does not change actions.

## Agent

Path:

`outputs/fuel_support_v2_from_best_agent`

Additional config:

- `fuel_support_adjacent_dry_run: True`
- `fuel_support_adjacent_city_fuel_turns_lt: 5.0`
- `fuel_support_adjacent_min_cargo_fuel: 20.0`

## Stability Check

### Fixed Smoke Seed

Fresh output:

`outputs/diagnostic_layer/fuel_support_v2_seed1259068876_16_fresh`

Seed:

`1259068876`

Result:

- vs first: win_rate 1.0, effective_survival_rate 1.0
- vs stage400: win_rate 0.5, effective_survival_rate 1.0
- vs v4_stage350: win_rate 1.0, effective_survival_rate 1.0
- shadow_safe: true

This matches the best-agent and fuel_support_v1 results on the same seed.

### Random 3-Seed v1 Stability Attempt

Output:

`outputs/fuel_support_v1_from_best_agent/evaluation_random3_16`

Seeds:

- `897501966`
- `330175776`
- `1861979770`

The run timed out on the ninth replay:

`map_16x16_vs_stage400_330175776_p0.json`

The first 8 completed replays were summarized:

- games: 8
- win_rate: 0.375
- effective_survival_rate: 0.875
- survival_rate: 0.875
- mean_city_tiles: 26.25
- worst_night_city_loss: 11

This did not show catastrophic behavior like the previous direct `bw` gate experiments, but the sample is incomplete because of the timeout.

## Interpretation

Fuel support remains much safer than action blocking. The current v2 does not yet improve aggregate metrics, but it creates the right logging path for the next step:

- Identify how often adjacent support opportunities appear.
- Check whether missed adjacent-support opportunities precede city loss.
- Only then consider a narrow movement override.

## Next Step

The evaluator does not preserve agent stderr dry-run logs. It creates `*.commands.json.stderr.log`, but those files are empty for agent debug output. Therefore adjacent support diagnostics must be mined offline from replay state and command JSON.

Offline miner:

`scripts/mine_adjacent_fuel_support.py`

The miner reads stateful replay files and the embedded/paired command stream, then detects:

- low-fuel friendly city tile near night/night
- adjacent cargo-carrying worker
- worker did not move onto that city tile
- future 5/10/20 turn team city loss

## Offline Miner Results

### v2 fixed smoke seed

Input:

`outputs/diagnostic_layer/fuel_support_v2_seed1259068876_16_fresh/replays`

Output:

`outputs/diagnostic_layer/fuel_support_v2_seed1259068876_16_fresh/adjacent_fuel_support_events.csv`

Summary:

- rows: 170
- missed_rows: 166
- supporting_rows: 4
- missed future_loss_10_rate: 0.5964
- missed mean_future_loss_10: 1.8976

### v1 random partial evaluation

Input:

`outputs/fuel_support_v1_from_best_agent/evaluation_random3_16/replays`

Output:

`outputs/fuel_support_v1_from_best_agent/evaluation_random3_16/adjacent_fuel_support_events.csv`

Summary:

- rows: 423
- missed_rows: 410
- supporting_rows: 13
- missed future_loss_10_rate: 0.7732
- missed future_big_loss_10_rate: 0.2439
- missed mean_future_loss_10: 3.4244

This is a stronger diagnostic signal than direct `bw` blocking, but it remains correlational. A movement override should start as a narrow dry-run candidate, not a battle rule.
