# GPU 32x32 Training Notes

This project is configured for local GPU training on a 32x32 Lux AI 2021 map.

## Verified Local Hardware

`nvidia-smi` detected:

- GPU: NVIDIA GeForce RTX 5060 Ti
- VRAM: 16 GB
- Driver CUDA support: 13.2

The project virtual environment was updated from CPU-only PyTorch to the
official CUDA 13.2 wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall `
  torch==2.12.1 `
  --index-url https://download.pytorch.org/whl/cu132
```

Validation command:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Verified result:

```text
2.12.1+cu132
True
13.2
NVIDIA GeForce RTX 5060 Ti
```

PyTorch source: https://pytorch.org/get-started/previous-versions/

## Code Change For 32x32 Maps

The original training entry point did not pass arbitrary Lux engine
configuration through Hydra. These files now allow map overrides:

- `run_monobeast.py`: adds default `env_configuration={}`.
- `lux_ai/lux_gym/__init__.py`: passes `flags.env_configuration` to `LuxEnv`.
- `lux_ai/lux_gym/lux_env.py`: merges custom config into the official default
  Lux AI 2021 configuration instead of replacing it.

The environment was verified with:

```powershell
$env:PATH="C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:PATH"
.\.venv\Scripts\python.exe -c "from lux_ai.lux_gym.lux_env import LuxEnv; from lux_ai.lux_gym.act_spaces import BasicActionSpace; from lux_ai.lux_gym.obs_spaces import FixedShapeContinuousObsV2; env=LuxEnv(BasicActionSpace(), FixedShapeContinuousObsV2(), configuration={'width':32,'height':32,'loglevel':0}, seed=1); obs, reward, done, info = env.reset(); print('map', obs.map_width, obs.map_height); env.close()"
```

Verified result:

```text
map 32 32
```

## Verified GPU Smoke Run

This command successfully completed a short 32x32 GPU training run:

```powershell
$env:PATH="C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:PATH"
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py `
  --config-name conv_config `
  actor_device=cuda:0 `
  learner_device=cuda:0 `
  disable_wandb=True `
  use_teacher=False `
  +use_mixed_precision=True `
  sharing_strategy=file_system `
  +env_configuration.width=32 `
  +env_configuration.height=32 `
  +env_configuration.loglevel=0 `
  num_actors=1 `
  n_actor_envs=1 `
  batch_size=1 `
  unroll_length=4 `
  total_steps=32 `
  +checkpoint_freq=1 `
  model_log_freq=1 `
  n_blocks=2 `
  hidden_dim=64 `
  embedding_dim=16
```

Run output:

- Output directory: `outputs/07-21/22-57-04`
- Checkpoint: `outputs/07-21/22-57-04/32.pt`
- Weights: `outputs/07-21/22-57-04/32_weights.pt`
- Model size: 451,881 parameters
- Final line: `Learning finished after 32 steps`
- Speed observed: 6.4 SPS / 1.6 BPS

## Next Larger Local Run

After the smoke run, a reasonable first longer 32x32 GPU experiment is:

```powershell
$env:PATH="C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:PATH"
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py `
  --config-name conv_config `
  actor_device=cuda:0 `
  learner_device=cuda:0 `
  disable_wandb=True `
  use_teacher=False `
  +use_mixed_precision=True `
  sharing_strategy=file_system `
  +env_configuration.width=32 `
  +env_configuration.height=32 `
  +env_configuration.loglevel=0 `
  num_actors=2 `
  n_actor_envs=1 `
  batch_size=2 `
  unroll_length=8 `
  total_steps=10000 `
  +checkpoint_freq=1000 `
  model_log_freq=100 `
  n_blocks=4 `
  hidden_dim=128 `
  embedding_dim=32
```

Scale gradually after this works:

- Increase `total_steps` first.
- Then increase `num_actors` and `batch_size`.
- Then increase `n_blocks`, `hidden_dim`, and `embedding_dim`.

Watch GPU memory with:

```powershell
nvidia-smi -l 2
```

## Completed 10000-Step Run

The larger local run above was executed successfully on 2026-07-21:

- Output directory: `outputs/07-21/23-01-26`
- Checkpoint: `outputs/07-21/23-01-26/10000.pt`
- Weights: `outputs/07-21/23-01-26/10000_weights.pt`
- Model size: 3,437,609 parameters
- Final line: `Learning finished after 10000 steps`
- Final checkpoint line: `Saving checkpoint to 10000`
- Typical observed speed: about 54-58 SPS / 3.4-3.6 BPS

This run used:

```text
width=32
height=32
actor_device=cuda:0
learner_device=cuda:0
num_actors=2
batch_size=2
unroll_length=8
total_steps=10000
n_blocks=4
hidden_dim=128
embedding_dim=32
```

## Replay From The Trained Model

Training checkpoints are not visualizer inputs. To visualize a trained model,
first package the checkpoint as an agent and run one Lux match to produce replay
JSON.

The local evaluation agent for the 10000-step model is:

```text
local_agents/trained_32x32_10000/main.py
```

It contains:

- `lux_ai/rl_agent/config.yaml`: copied from `outputs/07-21/23-01-26/config.yaml`
- `lux_ai/rl_agent/10000_weights.pt`: copied from
  `outputs/07-21/23-01-26/10000_weights.pt`
- `lux_ai/rl_agent/rl_agent_config.yaml`: uses no test-time data augmentation
  so replay generation is faster

Regenerate the replay with:

```powershell
pnpm run replay:trained-gpu32
```

The generated replay is:

```text
replays/trained_32x32_10000_vs_public_seed12345.json
```

Verified replay metadata:

- Width: 32
- Height: 32
- Seed: 12345
- Stateful turns: 157
- Command turns: 156
- Size: 3,164,688 bytes
- Result: public `working_title_bot_tong_hui_kang` rank 1, trained local agent
  rank 2

Upload this JSON to:

```text
https://2021vis.lux-ai.org/
```
