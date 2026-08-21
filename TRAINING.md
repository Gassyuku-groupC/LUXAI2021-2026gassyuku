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

## Progressive Backbone Migration

The retained 8, 16, and 24-block configs define the progressive depth route. Use:

```powershell
.\.venv\Scripts\python.exe .\scripts\migrate_progressive_resnet_checkpoint.py --help
```

Migration copies matching legacy keys and initializes only newly added residual blocks. Always verify checkpoint loading and Step-0 behavior before training the deeper model.
