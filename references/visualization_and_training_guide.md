# Visualization and Training Guide

This file describes the current training, packaging, replay, and visualization workflow. The project is based on `IsaiahPressman/Kaggle_Lux_AI_2021`, but this guide focuses on our local group workflow.

## Official Visualizer

Lux AI 2021 official visualizer:

```text
https://2021vis.lux-ai.org/
```

Usage:

1. Open the visualizer page.
2. Upload a replay JSON file.
3. Select the current replay:

```text
replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json
```

4. Use the timeline, zoom, statistics, and debug annotations to inspect agent behavior.

## Local Visualizer

Official local visualizer project:

```text
https://github.com/Lux-AI-Challenge/LuxViewer2021
```

Basic workflow:

```powershell
npm i -g serve
serve dist
```

Then open:

```text
http://localhost:5000
```

If using this repository's Node dependencies, install them first:

```powershell
pnpm install
```

## Generating Replays

The Lux AI 2021 CLI comes from the npm package `@lux-ai/2021-challenge`. This repository manages it through `package.json` and `pnpm-lock.yaml`.

Example command:

```powershell
.\node_modules\.bin\lux-ai-2021.CMD `
  local_agents\teacher_finetune_16x16_100000\main.py `
  internal_testing\public_agents\working_title_bot_tong_hui_kang\main.py `
  --python ".\.venv\Scripts\python.exe" `
  --loglevel 2 `
  --memory 8000 `
  --maxtime 20000 `
  --width 16 `
  --height 16 `
  --seed 12345 `
  --storeLogs=true `
  --statefulReplay=true `
  --out replays\teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json
```

`--statefulReplay=true` is important because it stores richer map state, making it easier to inspect resources, units, cities, and fuel changes in the visualizer.

## Training Entry Point

Main training entry:

```text
run_monobeast.py
```

Internal path:

```text
run_monobeast.py
  -> lux_ai/torchbeast/monobeast.py
  -> lux_ai/lux_gym/__init__.py
  -> lux_ai/lux_gym/lux_env.py
  -> official lux-ai-2021 engine
```

Current main config:

```text
conf/conv_teacher_finetune_16x16.yaml
```

Run example:

```powershell
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py --config-name conv_teacher_finetune_16x16
```

## Current Training Idea

Current route:

```text
1st place teacher/reference
  -> 16x16 imitation / teacher KL
  -> self-play finetune
  -> checkpoint every 10000 steps
  -> package agent
  -> replay validation
  -> visualizer inspection
```

Training priorities:

- Do not optimize only for early score. First make units and cities survive longer.
- Stabilize early wood collection for the first and second nights.
- Research coal and uranium when city tile count allows it.
- Expand cities gradually without letting fuel upkeep grow out of control.
- Inspect whether workers spread out, avoid congestion, and return resources to cities.

## Map Size Strategy

Legal Lux AI 2021 map sizes include:

```text
12x12, 16x16, 24x24, 32x32
```

We currently start from 16x16 because:

- It has more resources and expansion space than 12x12.
- It is faster to train than 24x24 or 32x32.
- It is a good middle ground for validating imitation learning and self-play finetuning.

Future configs:

```text
conf/conv_teacher_finetune_24x24.yaml
conf/conv_teacher_finetune_32x32.yaml
conf/conv_teacher_finetune_random_sizes.yaml
```

These should be used to test how map size changes the learned strategy.

## Outputs and Cleanup

Training outputs are written to:

```text
outputs/
```

This directory is ignored by `.gitignore`. We keep useful final checkpoints and logs locally, but do not push large intermediate training outputs to GitHub.

Final agents are stored in:

```text
local_agents/
```

Current replays are stored in:

```text
replays/
```
