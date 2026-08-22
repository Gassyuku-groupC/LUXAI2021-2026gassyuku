# Training

Commands assume PowerShell and the project root `D:\Luxai\Kaggle_Lux_AI_2021`.

## Local Artifacts

The repository does not include weights or replays. Expected local inputs are:

```text
outputs/submission_packages/best_agent/
internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/
dataset/raw/data/
replays/first battle/
```

The configured student starts from the historical first-place Actor. The immutable KL teacher is `outputs/submission_packages/best_agent/lux_ai/rl_agent/candidate_weights.pt`.

## 1. Generate Target-Policy Replays

```powershell
Set-Location D:\Luxai\Kaggle_Lux_AI_2021
Set-ExecutionPolicy -Scope Process Bypass

.\scripts\generate_deployed_agent_replays.ps1 `
  -CurrentAgent .\outputs\current_agent `
  -MapSizes 12,16,24,32 `
  -OpponentNames best_agent,first,stage350,stage400 `
  -Sides 0,1 `
  -ContinueOnFailure
```

The 32-map set may be generated separately because long games can exceed the replay timeout. A timeout is recorded as a failed replay and must not be treated as a valid sample.

## 2. Build Replay-Groups And Shards

```powershell
.\.venv\Scripts\python.exe .\scripts\build_imitation_index.py --help
.\.venv\Scripts\python.exe .\scripts\extract_imitation_shards.py --help
```

Select raw and deployed replay directories explicitly. Keep every replay/seed group in exactly one of train, validation, or calibration. Do not randomly split frames.

## 3. Train And Calibrate The Spatial Sidecar

```powershell
.\.venv\Scripts\python.exe .\scripts\train_spatial_risk_sidecar.py --help
.\.venv\Scripts\python.exe .\scripts\calibrate_spatial_risk_sidecar.py --help
```

Run calibration independently for 12, 16, 24, and 32 maps. Verify that each report includes sample counts, a PR curve, and a precision >= 0.85 threshold. Down-weight the small deployed 32-map set and retain raw replay groups as its primary calibration support.

## 4. Verify Step-0 Equivalence

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_step0_equivalence.py --help
```

The verification must pass all four map sizes, preserve the base Actor state hash, keep maximum initial logit difference at or below `1e-7`, and produce 100% matching selected actions.

## 5. Actor + Sidecar Behavior Cloning

```powershell
.\.venv\Scripts\python.exe .\scripts\train_imitation_bc.py --help
```

Use grouped training and validation shards. Monitor terminal metrics plus `metrics.csv`, `metrics.jsonl`, and recovery checkpoints. Increase batch size to 32 or 64 only after confirming GPU memory and validation stability. Invalid expert targets are expected when replay actions cannot be represented by the reconstructed legal mask; they are filtered and counted, not silently optimized.

The current APPO config expects the converged BC checkpoint at:

```text
outputs/first_actor_sidecar_bc_v3/best.pt
```

## 6. KL-APPO Smoke Run

```powershell
.\.venv\Scripts\python.exe .\run_monobeast.py `
  --config-name conv_sidecar_appo_vtrace `
  total_games=200 2>&1 | Tee-Object .\appo_v3_smoke_fixed.log
```

Successful completion requires:

- a `Learning finished` line;
- no `Non-finite`, `loss INVALID`, or learner traceback;
- a final full checkpoint and weights-only checkpoint;
- finite Actor, Gate, auxiliary-head, optimizer, and scheduler state;
- changed learned parameters compared with the BC checkpoint.

The shutdown-only `CudaIPCTypes.cpp` warning is not a training failure when it appears after `Learning finished`. PowerShell wraps native stderr as `NativeCommandError`, which can make this warning look fatal.

The verified 200-game run completed 70,560 steps. Actor and Gate weights changed and remained finite. Sidecar weights did not change because APPO deliberately detaches calibrated risk outputs before the Gate.

## 7. Longer Training And Evaluation

Do not promote the 200-game checkpoint directly. First package it and generate paired fixed-seed replays on 12, 16, 24, and 32 maps from both player positions. Compare against `best_agent`, first, stage350, and stage400.

PFSP pool sampling can be inspected with:

```powershell
.\.venv\Scripts\python.exe .\scripts\sample_pfsp_opponents.py `
  --pool-config .\conf\league_pool_sidecar.json `
  --count 1000 `
  --output .\outputs\pfsp_schedule.json
```

Continue low-learning-rate APPO only if KL remains controlled and paired evaluation does not regress expansion tempo, side balance, worst-night city loss, or win rate.

## 8. Role Adapter Evaluation And Joint Training

Generate the role-enabled package with `prepare_checkpoint_agents.py
--enable-role-adapter`, then compare it with `best_agent` on fixed seeds, both
sides, and all four maps. Treat 32x32 engine timeouts as failed evaluations, not
losses or valid samples. Review per-map results instead of relying only on the
aggregate ranking.

After fresh-seed replication, training may progress from adapter-only updates to
Role Adapter plus Sidecar, and finally to low-learning-rate joint Actor,
Sidecar, and Role Adapter optimization. Keep the frozen `best_agent` KL teacher
through every stage and use separate optimizer parameter groups. The learner
must explicitly save and restore adapter parameters and optimizer state before
learned-role training begins. See [ROLE.md](ROLE.md) for the interface and
initial learning-rate ranges.

### Role-only Repair

The first repair stage freezes Actor, spatial Sidecar, and intervention Gate and
optimizes exactly 14 role bias scalars. It uses a fixed best-agent KL teacher,
keeps teacher BC at `0.05`, samples maps with weights `12:16:24 = 1:2:3`, and
up-weights the pre-night/night window on 16x16 and 24x24.

```powershell
Set-Location D:\Luxai\Kaggle_Lux_AI_2021

.\.venv\Scripts\python.exe .\run_monobeast.py `
  --config-name conv_role_only_repair 2>&1 |
  Tee-Object .\outputs\role_only_repair\train.log
```

Export a selected checkpoint into the fixed runtime adapter format:

```powershell
.\.venv\Scripts\python.exe .\scripts\export_role_bias_checkpoint.py `
  --checkpoint .\outputs\role_only_repair\CHECKPOINT_weights.pt `
  --output .\outputs\role_only_repair\role_city_bias_params.yaml
```

Do not promote from loss alone. Package the exported YAML and repeat paired
12/16/24 evaluation. Require unchanged Actor/Sidecar/Gate tensors, finite role
parameters, controlled Teacher KL, and lower paired 24x24 night-loss delta.

### 70560 Rescue Stage

The first checkpoint screen promoted 70560 but exposed small-map and player-1
regressions. Run the bounded rescue stage from its weights with a fresh optimizer:

```powershell
$out = "D:\Luxai\Kaggle_Lux_AI_2021\outputs\sidecar_appo_rescue_70560"
New-Item -ItemType Directory -Path $out -Force | Out-Null
Push-Location $out
& ..\..\.venv\Scripts\python.exe ..\..\run_monobeast.py `
  --config-name conv_sidecar_appo_rescue_70560 2>&1 |
  Tee-Object .\train.log
Pop-Location
```

This stage runs 100 games at `1e-6`, raises the fixed-best reference KL cost to
`0.01`, retains a `0.05` teacher-BC floor, and samples 12x12/16x16 in four of
six map slots. Evaluate its final weights immediately on paired 12x12/24x24
games. Stop APPO if neither small-map nor player-1 performance improves.

## Checkpoint Selection

Build executable packages for the BC starting point and every checkpoint from the
200-game smoke run:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_checkpoint_agents.py
```

Run phase 1 against `best_agent` on 12x12 and 24x24 maps from both sides:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_checkpoint_selection.ps1 `
  -Phase phase1 `
  -Seeds 20260824 `
  -AgentTurnTimeoutMs 30000 `
  -TimeoutSeconds 1200 `
  -SkipPackaging
```

The controller reuses valid replays, so it can be stopped and resumed. After
phase 1, pass only the promoted checkpoint labels to phase 2:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_checkpoint_selection.ps1 `
  -Phase phase2 `
  -Checkpoints 30272,50112,70560 `
  -Seeds 20260824 `
  -AgentTurnTimeoutMs 30000 `
  -TimeoutSeconds 1800 `
  -SkipPackaging
```

Phase 2 evaluates all four map sizes against `best_agent`, first, stage350, and
stage400. Results are written to `ranking.csv`, `games.csv`, and `summary.json`.
Ranking is ordered by win rate, timeout rate, city margin, unit margin, and
worst-night city loss. Training loss is not a promotion criterion.

## Progressive Backbone Migration

The retained 8, 16, and 24-block configs define the progressive depth route. Use:

```powershell
.\.venv\Scripts\python.exe .\scripts\migrate_progressive_resnet_checkpoint.py --help
```

Migration copies matching legacy keys and initializes only newly added residual blocks. Always verify checkpoint loading and Step-0 behavior before training the deeper model.
