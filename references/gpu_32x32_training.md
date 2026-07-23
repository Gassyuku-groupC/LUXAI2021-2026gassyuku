# GPU and Large Map Training Notes

本文件记录 GPU 训练和大地图扩展路线。当前主线仍以 16x16 为起点，32x32 属于后续扩展目标。

## 本地 GPU

本地机器已验证可使用 NVIDIA GPU 进行 PyTorch 训练。检查命令：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

如果 `torch.cuda.is_available()` 返回 `True`，即可使用：

```yaml
actor_device: cuda:0
learner_device: cuda:0
```

## 地图配置传递

原项目训练入口没有直接把任意 Lux engine configuration 从 Hydra 传到环境中。本仓库保留了以下本地改动：

- `run_monobeast.py`: 增加默认 `env_configuration={}`。
- `lux_ai/lux_gym/__init__.py`: 创建环境时传入 `flags.env_configuration`。
- `lux_ai/lux_gym/lux_env.py`: 先读取官方默认配置，再合并自定义 `width`、`height`、`loglevel`。

因此可以在 YAML 中写：

```yaml
env_configuration:
  width: 16
  height: 16
  loglevel: 0
```

也可以在命令行覆盖：

```powershell
+env_configuration.width=32 +env_configuration.height=32
```

## 当前主训练

当前主训练不是 32x32，而是：

```text
conf/conv_teacher_finetune_16x16.yaml
```

目标：

- 100000 steps 小规模验证。
- 每 10000 learner steps 保存 checkpoint。
- 使用 teacher imitation / teacher KL 稳定策略。
- 通过 replay 检查生存、扩张、研究和采矿行为。

## 扩展到 24x24 / 32x32

后续扩展顺序建议：

1. 继续在 16x16 上修正奖励和行为问题。
2. 跑 24x24 小规模训练，观察是否能保持扩张和 fuel 管理。
3. 跑 32x32 小规模训练，观察 worker 是否能有效分散。
4. 最后使用 `conv_teacher_finetune_random_sizes.yaml` 混合地图尺寸训练。

对应配置：

```text
conf/conv_teacher_finetune_24x24.yaml
conf/conv_teacher_finetune_32x32.yaml
conf/conv_teacher_finetune_random_sizes.yaml
```

## 32x32 示例命令

```powershell
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py `
  --config-name conv_teacher_finetune_32x32 `
  actor_device=cuda:0 `
  learner_device=cuda:0 `
  disable_wandb=True
```

如果显存不足，可以优先降低：

- `num_actors`
- `n_actor_envs`
- `batch_size`
- `unroll_length`
- `n_blocks`
- `hidden_dim`

## 观察指标

大地图训练时不要只看训练是否结束，还要看 replay：

- 360 turn 是否能稳定存活。
- 是否只集中在一个 wood 点，还是能扩张到多个资源区。
- research 是否能到 50 和 200。
- city fuel 是否能覆盖夜晚 upkeep。
- worker 是否拥堵或空转。
- 对手压力下是否还能保留城市和 worker。

这些指标决定是否继续放大训练步数。
