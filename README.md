# LUXAI2021-2026gassyuku

This repository is the Lux AI Season 1 / Lux AI 2021 experiment workspace for our group project. The current work does not rebuild the environment from scratch. Instead, it builds on the open-source first-place solution, then adds a local workflow for environment setup, GPU training, agent packaging, replay generation, and visualization.

## Project Origin

This project is based on Isaiah Pressman's first-place Lux AI 2021 repository:

- Original repository: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- Kaggle 1st place write-up: https://www.kaggle.com/c/lux-ai-2021/discussion/294993

The original repository provides a TorchBeast/IMPALA-style self-play reinforcement learning framework, Lux AI environment wrappers, neural network models, historical first-place agents, and replay analysis resources. This repository reorganizes that foundation for our group's local experiments while preserving attribution and the original license file.

## Current Goal

The current goal is to make our agent survive longer in Lux AI 2021 and gradually learn the balance between city expansion, mining, research, and self-play strategy from the first-place solution. Our priorities are:

1. Keep the full training, packaging, replay, and visualization pipeline reproducible.
2. Use the first-place agent as a teacher for imitation learning and self-play finetuning.
3. Start from 16x16 maps, then extend to 24x24, 32x32, and mixed map sizes.
4. Prioritize surviving to 360 turns before optimizing win rate and final score.

## Verified Current Route

We have completed one small-scale 16x16 teacher-finetuning run:

- Map size: 16x16
- Training steps: 100000
- Checkpoint interval: every 10000 learner steps
- Final weights: `100000_weights.pt`
- Packaged agent: `local_agents/teacher_finetune_16x16_100000.zip`
- Validation replay: `replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json`

In the validation replay, the current agent defeated a public/reference opponent on a 16x16 map and expanded into multiple cities, many workers, and full uranium research. This result is the baseline for future research.

## Repository Structure

```text
conf/
  conv_teacher_finetune_16x16.yaml        Main current training config
  conv_teacher_finetune_24x24.yaml        Future 24x24 map config
  conv_teacher_finetune_32x32.yaml        Future 32x32 map config
  conv_teacher_finetune_random_sizes.yaml Future mixed-size map config

lux_ai/
  lux/                                    Lux AI 2021 game objects and rules
  lux_gym/                                Gym environment, action spaces, observation spaces, rewards
  nns/                                    Neural network models
  rl_agent/                               Agent inference code
  torchbeast/                             IMPALA/TorchBeast training loop

internal_testing/
  hall_of_fame/                           Strong agents and teacher references from the base project
  public_agents/                          Public/reference agents

local_agents/
  teacher_finetune_16x16_100000/          Current packaged agent source directory
  teacher_finetune_16x16_100000.zip       Current uploadable/submittable agent package

replays/
  teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json

references/
  kaggle_lux_ai_2021_top_results.md       Kaggle top-solution references
  replay_validation_1st_place.md          Replay validation notes
  visualization_and_training_guide.md     Training and visualization workflow
  gpu_32x32_training.md                   GPU and large-map training notes
```

`outputs/`, `.venv/`, and `node_modules/` are local training outputs and dependency directories. They are ignored by `.gitignore` and are not pushed as repository content.

## Environment

On Windows, the recommended workflow is to use the local virtual environment from PowerShell:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
pnpm install
```

The official Lux AI CLI depends on Node.js. This repository uses `package.json` and `pnpm-lock.yaml` to pin the replay/visualization-related JavaScript dependencies.

Docker can also be used:

```powershell
docker compose build
docker compose run --rm luxai powershell
```

## Training

Current main config:

```powershell
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py --config-name conv_teacher_finetune_16x16
```

Training entry path:

```text
run_monobeast.py
  -> lux_ai/torchbeast/monobeast.py
  -> lux_ai/lux_gym/LuxEnv
  -> official lux-ai-2021 engine
```

Local changes kept in this repository:

- Hydra configs can pass `env_configuration.width` and `env_configuration.height`.
- Checkpoints are saved by learner-step interval instead of elapsed minutes.
- Training logs are quieter by suppressing Gym/Hydra/CUDA warning noise.
- `run_monobeast.py` does not automatically resume from a local `config.yaml` unless explicitly requested.

## Agent Packaging

The current usable agent package is:

```text
local_agents/teacher_finetune_16x16_100000.zip
```

The zip was rebuilt from:

```text
local_agents/teacher_finetune_16x16_100000/
```

`__pycache__` files were removed before packaging. The main model weights are:

```text
local_agents/teacher_finetune_16x16_100000/lux_ai/rl_agent/100000_weights.pt
```

## Replay and Visualization

Current replay:

```text
replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json
```

It can be uploaded to the official Lux AI 2021 visualizer:

```text
https://2021vis.lux-ai.org/
```

The official local visualizer project is:

```text
https://github.com/Lux-AI-Challenge/LuxViewer2021
```

More commands and details are in `references/visualization_and_training_guide.md`.

## Next Steps

- Continue studying the first-place and other top solutions to improve city expansion, research timing, and fuel management.
- Stabilize behavior on 16x16 before moving to 24x24 and 32x32.
- Add a more systematic evaluation script across checkpoints, map sizes, seeds, and opponents.
- After survival to 360 turns becomes more stable, optimize win rate, city count, and final score.

## Attribution

This project is based on `IsaiahPressman/Kaggle_Lux_AI_2021`. The original repository, training framework, model architecture, and many reference agents were created by Isaiah Pressman and contributors. This group project keeps that attribution while reorganizing the repository for local experiments, teacher finetuning, replay validation, and future group research.
