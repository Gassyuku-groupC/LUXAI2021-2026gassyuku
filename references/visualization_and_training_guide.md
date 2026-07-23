# Lux AI 2021 Visualization and Training Guide

This guide is for the imported 1st place solution repository:

https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021

## Replay Visualization

### Online Visualizer

The official Season 1 visualizer is hosted here:

https://2021vis.lux-ai.org/

Use it like this:

1. Open the visualizer URL in a browser.
2. Click or drag into "Upload a replay".
3. Select a replay JSON, for example:
   `replays/first_place_validation_12x12.json`
4. Use play/pause, timeline scrubbing, zoom, pan, stats, warnings, and debug
   annotation toggles to inspect the game.

The current validation replay was generated with `--statefulReplay=true`, so it
contains richer map state than the minimal action-only replay.

### Local Visualizer

The visualizer source/release project is:

https://github.com/Lux-AI-Challenge/LuxViewer2021

The documented local workflow is:

1. Download a release from `LuxViewer2021`.
2. Unzip it; it should contain a `dist` directory.
3. Serve `dist` locally.

With npm:

```powershell
npm i -g serve
serve dist
```

Then open:

```text
http://localhost:5000
```

Higher quality rendering can be requested by appending `?scale=2`; lower quality
can use `?scale=1`.

## Generating More Replays

The official Season 1 engine/CLI is `lux-ai-2021`. In this project it is
installed locally through `@lux-ai/2021-challenge@3.1.1`.

Set PATH for the current PowerShell session:

```powershell
$env:PATH="C:\Users\YE ZIHAN\.venvs\lux-ai-2021\Scripts;C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:PATH"
```

Generate a 12x12 stateful replay:

```powershell
.\node_modules\.bin\lux-ai-2021.CMD `
  internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research\main.py `
  internal_testing\public_agents\working_title_bot_tong_hui_kang\main.py `
  --python "C:\Users\YE ZIHAN\.venvs\lux-ai-2021\Scripts\python.exe" `
  --loglevel 2 `
  --memory 8000 `
  --maxtime 20000 `
  --width 12 `
  --height 12 `
  --storeLogs=true `
  --statefulReplay=true `
  --out replays\my_validation_12x12.json
```

## Training Overview

The repository trains through:

```text
run_monobeast.py -> lux_ai/torchbeast/monobeast.py -> lux_ai/lux_gym/LuxEnv
```

The algorithm is a TorchBeast/IMPALA-style self-play loop with V-trace, UPGO,
TD-lambda, optional teacher KL, and Hydra YAML configs under `conf/`.

Important local constraints:

- `LuxEnv` launches `node .../dimensions/main.js`, so `node` must be on PATH.
- The original configs assume CUDA devices like `cuda:0` and `cuda:1`.
- `resume_config.yaml` contains original author absolute paths; do not use it
  unchanged locally.
- For local first checks, disable W&B and use CPU/small settings.
- Full training is expensive; the 1st place write-up describes multi-GPU,
  overnight training.

## Training Smoke Test

Run this first to verify the pipeline, not to get a strong model:

```powershell
$env:PATH="C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:PATH"
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py `
  --config-name conv_config `
  actor_device=cpu `
  learner_device=cpu `
  disable_wandb=True `
  use_teacher=False `
  +use_mixed_precision=False `
  sharing_strategy=file_system `
  num_actors=1 `
  n_actor_envs=1 `
  batch_size=1 `
  unroll_length=2 `
  total_steps=4 `
  +checkpoint_freq=1 `
  model_log_freq=1 `
  n_blocks=1 `
  hidden_dim=16 `
  embedding_dim=8
```

Expected output:

- Hydra creates an `outputs/<date>/<time>/` directory.
- A `config.yaml` appears in that run directory.
- Checkpoints are saved periodically as `*.pt`.
- The validated local smoke test reached `Learning finished after 4 steps` and
  saved checkpoint `4`.

If this fails on Windows multiprocessing or Node subprocess behavior, use WSL2
or the Docker path below.

## Training With GPU

For the verified local CUDA 13.2 / 32x32 setup, see:

```text
references/gpu_32x32_training.md
```

Once the smoke test works, use a real config and override local paths/devices.
For example, phase 1 shaped reward:

```powershell
$env:PATH="C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:PATH"
.\.venv\Scripts\python.exe run_monobeast.py `
  --config-name conv_phase1_shaped_reward `
  actor_device=cuda:0 `
  learner_device=cuda:0 `
  disable_wandb=True `
  +use_mixed_precision=True `
  sharing_strategy=file_system
```

If `conv_phase1_shaped_reward.yaml` raises
`FixedShapeContinuousObsV2() takes no arguments`, remove or adjust
`obs_space_kwargs.include_subtask_encoding` for that config before using it for
local training. The generic `conv_config.yaml` smoke test above avoids this
issue.

For later phases with a teacher model:

1. Pick a previous run directory under `outputs/`.
2. Set `teacher_load_dir=<that-run-dir>`.
3. Set `teacher_checkpoint_file=<checkpoint>.pt`.
4. Keep `use_teacher=True`.

Avoid using the original `teacher_load_dir` values in the configs because they
point to the original author's machine.

## Recommended Experiment Order

1. `conv_phase1_shaped_reward.yaml`: small 8-block model with shaped rewards.
2. `conv_phase2_game_result.yaml`: transition toward game-result reward.
3. `conv_phase3_small_teacher.yaml`: train with a smaller teacher.
4. `conv_phase4_small_teacher.yaml`: continue scaling.
5. `conv_phase5+_final_model.yaml`: final larger model style.

Use tiny overrides for debugging, then gradually increase:

- `total_steps`
- `num_actors`
- `n_actor_envs`
- `batch_size`
- `n_blocks`

## Docker/WSL Recommendation

For serious training, prefer Linux via Docker or WSL2. The project Dockerfile
already installs Python dependencies and Node.js. Windows native training may
work for small smoke tests, but the original code path was designed around
Linux-like multiprocessing and long-running Node subprocesses.
