# Runtime Gate V1

## Purpose

Runtime Gate V1 is a conservative emergency brake for the current best agent.

It does not retrain the actor and does not replace the whole turn plan. It only
marks extreme `BUILD_WORKER` and worker `BUILD_CITY` candidates as illegal in
the existing action resolver, so the model's next-best candidate action is used.

Current runnable gated agent:

- `outputs/runtime_gate_v1_from_best/agent`

Current safe baseline:

- `outputs/auto_league_dagger_v10_shadow/best_agent`

## What It Blocks

The gate activates only when at least one of these is true:

- late game: `turn >= runtime_gate_v1_late_turn`
- night
- within `runtime_gate_v1_pre_night_turns` turns before night

Then it blocks:

- `BUILD_WORKER` when the acting city or team fuel buffer is extremely low
- `BUILD_CITY` when team fuel buffer is extremely low

On 32x32, the p25 fuel threshold is stricter after the agent already has a large
city scale. This targets the observed large-map pattern where one side expands
into a huge city count and then loses many city tiles near the end.

## Configuration

The gate is enabled in:

- `outputs/runtime_gate_v1_from_best/agent/lux_ai/rl_agent/rl_agent_config.yaml`

Important fields:

- `runtime_gate_v1_enabled`
- `runtime_gate_v1_late_turn`
- `runtime_gate_v1_pre_night_turns`
- `runtime_gate_v1_min_city_fuel_turns_block_bw`
- `runtime_gate_v1_p25_city_fuel_turns_block_bw`
- `runtime_gate_v1_min_city_fuel_turns_block_bcity`
- `runtime_gate_v1_p25_city_fuel_turns_block_bcity`
- `runtime_gate_v1_large_map_extra_strict`

To disable the gate, set:

```yaml
runtime_gate_v1_enabled: False
```

## Smoke Test

The first smoke test completed normally:

- output: `outputs/diagnostic_layer/runtime_gate_v1_smoke_public_16`
- opponent: `public_ilialar_risk_averse`
- map size: 16
- games: 2
- win rate: 1.0
- shadow safe: true

This is only a runtime sanity check, not promotion evidence.

## Recommended Next Evaluations

Run same-seed A/B against the baseline before using this as the battle agent:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\evaluate_agent_generalization.ps1 `
  -CandidateAgent .\outputs\runtime_gate_v1_from_best\agent `
  -OutputDir .\outputs\diagnostic_layer\runtime_gate_v1_strong_16_batch1 `
  -RandomSeedCount 2 `
  -EvaluationMapSizes 16 `
  -OpponentNames first,stage400 `
  -ReplayTimeoutSeconds 240 `
  -MaxReplayAttempts 1 `
  -ContinueOnReplayFailure
```

For 32x32, run one seed at a time:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_fresh_seed_dry_run_gate_v1_safe.ps1 `
  -CandidateAgent .\outputs\runtime_gate_v1_from_best\agent `
  -OutputDir .\outputs\diagnostic_layer\runtime_gate_v1_strong_32_seed1 `
  -RandomSeedCount 1 `
  -EvaluationMapSizes 32 `
  -OpponentNames first,stage400 `
  -ReplayTimeoutSeconds 900
```
