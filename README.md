# LUXAI2021-2026gassyuku

本仓库是小组项目使用的 Lux AI Season 1 / Lux AI 2021 实验仓库。当前路线不是从零重写环境，而是在第一名开源方案的训练框架上，完成本地环境搭建、GPU 训练、agent 打包、replay 生成和可视化验证。

## 项目来源

本项目基于 Isaiah Pressman 的 Lux AI 2021 第一名开源仓库：

- 原始仓库: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- Kaggle 1st place write-up: https://www.kaggle.com/c/lux-ai-2021/discussion/294993

原仓库提供了 TorchBeast/IMPALA 风格的自博弈强化学习框架、Lux AI 环境封装、神经网络结构、第一名历史 agent 和比赛复盘资料。本仓库在此基础上整理出适合小组继续实验的本地路线，并保留原许可证文件。

## 当前目标

当前阶段的目标是让 agent 在 Lux AI 2021 中稳定生存更久，并逐步学习第一名方案中的城市扩张、采矿、研究和自博弈策略。现阶段优先级如下：

1. 跑通训练、打包、replay、可视化的完整链路。
2. 使用第一名 agent 作为 teacher，进行模仿学习和自博弈微调。
3. 先在 16x16 地图上训练，后续扩展到 24x24、32x32 和随机地图尺寸。
4. 优先提升 360 turn 生存能力，再进一步优化胜率和得分。

## 当前已验证路线

已完成一次 16x16 teacher finetune 小规模训练：

- 训练地图: 16x16
- 训练步数: 100000
- checkpoint 间隔: 10000 learner steps
- 当前最终权重: `100000_weights.pt`
- 打包 agent: `local_agents/teacher_finetune_16x16_100000.zip`
- 验证 replay: `replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json`

验证结果中，当前 agent 在 16x16 replay 中击败 public/reference opponent，并成功扩张到多城市、多 worker、研究完成 uranium 的状态。这个结果作为后续研究的起点。

## 目录结构

```text
conf/
  conv_teacher_finetune_16x16.yaml        当前主训练配置
  conv_teacher_finetune_24x24.yaml        后续 24x24 地图配置
  conv_teacher_finetune_32x32.yaml        后续 32x32 地图配置
  conv_teacher_finetune_random_sizes.yaml 后续随机地图尺寸配置

lux_ai/
  lux/                                    Lux AI 2021 游戏对象和规则封装
  lux_gym/                                Gym 环境、动作空间、观测空间、奖励空间
  nns/                                    神经网络模型
  rl_agent/                               agent 推理代码
  torchbeast/                             IMPALA/TorchBeast 训练循环

internal_testing/
  hall_of_fame/                           原项目保留的强 agent 和 teacher 参考
  public_agents/                          public/reference agents

local_agents/
  teacher_finetune_16x16_100000/          当前已打包 agent 源目录
  teacher_finetune_16x16_100000.zip       当前可提交/可上传 agent 包

replays/
  teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json

references/
  kaggle_lux_ai_2021_top_results.md       Kaggle top solution 参考
  replay_validation_1st_place.md          第一名 agent 复盘验证
  visualization_and_training_guide.md     训练和可视化链路
  gpu_32x32_training.md                   GPU 和大地图训练笔记
```

`outputs/`、`.venv/`、`node_modules/` 是本地训练输出和依赖目录，已经加入 `.gitignore`，不会作为仓库内容推送。

## 环境

推荐在 Windows PowerShell 中使用本地虚拟环境。

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
pnpm install
```

Lux AI 官方 CLI 依赖 Node.js。当前仓库使用 `package.json` 和 `pnpm-lock.yaml` 固定 replay/可视化相关 JS 依赖。

如果使用 Docker，可以参考：

```powershell
docker compose build
docker compose run --rm luxai powershell
```

## 训练

当前主配置：

```powershell
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py --config-name conv_teacher_finetune_16x16
```

训练入口链路：

```text
run_monobeast.py
  -> lux_ai/torchbeast/monobeast.py
  -> lux_ai/lux_gym/LuxEnv
  -> official lux-ai-2021 engine
```

本仓库对原训练入口做了几处本地化调整：

- 支持通过 Hydra 配置传入 `env_configuration.width` 和 `env_configuration.height`。
- checkpoint 频率按 learner step 保存，而不是按分钟保存。
- 简化训练日志，减少 Gym/Hydra/CUDA 的噪声输出。
- 默认不从当前目录的 `config.yaml` 自动恢复，避免误读旧实验配置。

## Agent 打包

当前可用 agent 位于：

```text
local_agents/teacher_finetune_16x16_100000.zip
```

该 zip 是从 `local_agents/teacher_finetune_16x16_100000/` 重新打包得到，已经清理 `__pycache__`。其中核心权重为：

```text
local_agents/teacher_finetune_16x16_100000/lux_ai/rl_agent/100000_weights.pt
```

## Replay 和可视化

当前 replay：

```text
replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json
```

可上传到官方 Lux AI 2021 visualizer：

```text
https://2021vis.lux-ai.org/
```

也可以使用官方本地 visualizer 项目：

```text
https://github.com/Lux-AI-Challenge/LuxViewer2021
```

更多命令和说明见 `references/visualization_and_training_guide.md`。

## 后续计划

- 继续参考第一名及其他 top solution 的策略，改进城市扩张、研究节奏和燃料管理。
- 在 16x16 上稳定训练后，迁移到 24x24 和 32x32。
- 引入更系统的评估脚本，对不同 checkpoint、地图尺寸、seed 和 opponent 做 replay 评估。
- 在生存到 360 turn 更稳定后，再优化胜率、城市数量和最终得分。

## Attribution

This project is based on `IsaiahPressman/Kaggle_Lux_AI_2021`. The original repository, training framework, model architecture, and many reference agents were created by Isaiah Pressman and contributors. This group project keeps that attribution while reorganizing the repository for local experiments, teacher finetuning, replay validation, and future group research.
