# Replay Validation Notes

本文件记录 replay 验证思路。当前项目需要同时验证两类 replay：

1. 原第一名 agent 是否能在本地正常运行。
2. 当前小组训练出的 agent 是否能生成可上传到官方 visualizer 的 replay。

## 第一名 Agent 验证

原第一名 hall-of-fame agent 位于：

```text
internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/main.py
```

它来自本项目基础仓库 `IsaiahPressman/Kaggle_Lux_AI_2021`，用于 teacher/reference 对照。

早期本地验证方式：

```powershell
.\node_modules\.bin\lux-ai-2021.CMD `
  internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research\main.py `
  internal_testing\public_agents\working_title_bot_tong_hui_kang\main.py `
  --python ".\.venv\Scripts\python.exe" `
  --loglevel 2 `
  --memory 8000 `
  --maxtime 20000 `
  --width 12 `
  --height 12 `
  --storeLogs=true `
  --statefulReplay=true `
  --out replays\first_place_validation_12x12.json
```

该文件现在不再作为主路线保留，作用是确认原项目 agent 和官方 Lux CLI 能在本地跑通。

## 当前 Agent 验证

当前主 replay：

```text
replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json
```

对应 agent：

```text
local_agents/teacher_finetune_16x16_100000
local_agents/teacher_finetune_16x16_100000.zip
```

当前 replay 的意义：

- 验证 `100000_weights.pt` 已经成功打包成 agent。
- 验证 16x16 地图可以跑完整 match。
- 验证输出 JSON 可以上传到官方 visualizer。
- 作为后续 checkpoint 评估和策略对比的 baseline。

## 评估重点

后续 replay 不只看胜负，还要记录：

- agent 是否能活到 360 turn。
- city tile 数量和 city 数量增长是否健康。
- worker 是否分散采矿，还是卡在单一 wood 点。
- research 是否能到 50/200，是否能采 coal/uranium。
- 夜晚前 fuel buffer 是否足够。
- 单位是否拥堵、空转或突然大量消失。

这些指标比单次胜负更适合指导下一阶段训练。
