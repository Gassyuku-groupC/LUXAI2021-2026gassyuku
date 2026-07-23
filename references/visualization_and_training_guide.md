# Visualization and Training Guide

本文件说明当前仓库的训练、打包、replay 和可视化链路。项目基于 `IsaiahPressman/Kaggle_Lux_AI_2021`，但当前文档以本地小组实验路线为准。

## 官方可视化工具

Lux AI 2021 官方 visualizer：

```text
https://2021vis.lux-ai.org/
```

使用方式：

1. 打开网页。
2. 上传 replay JSON。
3. 选择当前 replay：

```text
replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json
```

4. 使用时间轴、缩放、统计信息和 debug annotation 检查 agent 行为。

## 本地 Visualizer

官方本地 visualizer 项目：

```text
https://github.com/Lux-AI-Challenge/LuxViewer2021
```

基本流程：

```powershell
npm i -g serve
serve dist
```

然后打开：

```text
http://localhost:5000
```

如果使用当前仓库的 Node 依赖，先运行：

```powershell
pnpm install
```

## 生成 Replay

Lux AI 2021 CLI 来自 npm 包 `@lux-ai/2021-challenge`，当前仓库通过 `package.json` 和 `pnpm-lock.yaml` 管理。

示例命令：

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

`--statefulReplay=true` 很重要，它会保存更完整的地图状态，方便 visualizer 检查资源、单位、城市和 fuel 变化。

## 训练入口

训练主入口：

```text
run_monobeast.py
```

内部链路：

```text
run_monobeast.py
  -> lux_ai/torchbeast/monobeast.py
  -> lux_ai/lux_gym/__init__.py
  -> lux_ai/lux_gym/lux_env.py
  -> official lux-ai-2021 engine
```

当前主配置：

```text
conf/conv_teacher_finetune_16x16.yaml
```

运行示例：

```powershell
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py --config-name conv_teacher_finetune_16x16
```

## 当前训练思路

当前路线是：

```text
1st place teacher/reference
  -> 16x16 imitation / teacher KL
  -> self-play finetune
  -> checkpoint every 10000 steps
  -> package agent
  -> replay validation
  -> visualizer inspection
```

训练优先目标：

- 不只追求早期得分，而是让单位和城市活得更久。
- 前期稳定采 wood，保证第一夜和第二夜燃料。
- 在 city tile 数量允许时研究 coal 和 uranium。
- 逐步扩张城市，但不要让 fuel upkeep 失控。
- 观察 worker 是否分散、是否拥堵、是否能把资源送回城市。

## 地图尺寸策略

Lux AI 2021 合法地图尺寸包括：

```text
12x12, 16x16, 24x24, 32x32
```

当前选择 16x16 作为起点，原因是：

- 比 12x12 有更多资源和扩张空间。
- 比 24x24/32x32 训练更快。
- 更适合先验证模仿学习和自博弈微调是否有效。

后续可以使用：

```text
conf/conv_teacher_finetune_24x24.yaml
conf/conv_teacher_finetune_32x32.yaml
conf/conv_teacher_finetune_random_sizes.yaml
```

逐步测试地图尺寸变化对策略的影响。

## 输出和清理

训练输出位于：

```text
outputs/
```

该目录已被 `.gitignore` 忽略。当前只在本地保留最终有用 checkpoint 和日志，不把大量中间训练记录推送到 GitHub。

最终 agent 放在：

```text
local_agents/
```

当前 replay 放在：

```text
replays/
```
