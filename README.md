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

## Experiment History

The detailed Japanese training log for v1-v7, including configurations, evaluation
results, failure analysis, and checkpoint decisions, is available at
[`docs/training_log_v1_v7_ja.md`](docs/training_log_v1_v7_ja.md).

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
.\.venv\Scripts\python.exe run_monobeast.py --config-name conv_survival_research_buffer2_finetune_16x16
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

### Automated self-play league

The league controller trains in shorter game-count stages and evaluates every candidate from
both player positions against the current best agent and the first-place agent.
The v2 controller defaults to 100 completed games in 25-game stages, using two
seeds. It starts the cumulative learner from the most stable v1 stage-100
checkpoint while retaining the previous `best_agent` as the champion:

```powershell
Set-Location D:\Luxai\Kaggle_Lux_AI_2021
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\auto_train_league.ps1
```

For a short external smoke run:

```powershell
.\scripts\auto_train_league.ps1 -TotalGames 20 -GamesPerStage 10 -Seeds 12345
```

Each stage writes its weights, stateful replay files, and `evaluation.json` under
`outputs/auto_league_dagger_v2_16x16/`. Stage directories use names such as
`game_stage_00025`. A candidate is promoted to `best_agent` only when
all evaluation games survive to turn 360, the largest night-time city loss is at
most six tiles, and the combined win rate is at least 50 percent. These thresholds
can be adjusted near the end of `scripts/auto_train_league.ps1`.

The training environment already uses the same policy for both players. The
league layer adds model selection: candidate versus current best prevents
regression, while candidate versus first place tracks the external performance
gap. Stop the controller with `Ctrl+C`; completed stage directories and the
current `best_agent` remain available.

### Teacher BC and online DAgger

The default league config is `conv_teacher_bc_dagger_v2_16x16`. On every state
visited by the learner, the first-place teacher supplies hard targets for the
complete worker, city-tile, and cart action spaces. The loss combines RL,
teacher KL, and action-space-balanced behavior-cloning cross entropy. Worker,
city-tile, and cart BC weights are `2.0`, `3.0`, and `0.5`; the BC cost anneals
from `20.0` to `2.0` over 1,000 completed games, and RL policy loss is scaled by
`0.1` during this imitation-heavy phase.

Fuel buffer targets are expressed in complete nights. A value of `2.0` now
means 20 survivable night turns, not two turns. The v2 reward checks the full
30-turn day before night and applies stronger penalties to unsafe expansion and
city-tile loss.

Compact logs report BC loss and teacher-action accuracy for worker (`W`), city
tile (`C`), and cart (`K`) at the current BC cost:

```text
Games 20/25 | steps 7808 | loss 18.42 | bc 12.31 W82/C96/K55 @ 19.64
```

The bundled Sazuma fourth-place imitation-learning notebook remains a useful
reference for replay dataset construction. This project extends that idea by
using the existing full action space and querying the teacher on learner-visited
states instead of limiting supervision to five worker actions.

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
