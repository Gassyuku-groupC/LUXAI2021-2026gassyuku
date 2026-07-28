# GPU and Large Map Training Notes

This file records the GPU training and large-map extension plan. The current main route still starts from 16x16. 32x32 training is a later expansion target.

## Local GPU

The local machine has been verified to support PyTorch GPU training. Check with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` returns `True`, the configs can use:

```yaml
actor_device: cuda:0
learner_device: cuda:0
```

## Passing Map Configuration

The original training entry point did not directly pass arbitrary Lux engine configuration from Hydra into the environment. This repository keeps the following local changes:

- `run_monobeast.py`: adds default `env_configuration={}`.
- `lux_ai/lux_gym/__init__.py`: passes `flags.env_configuration` when creating `LuxEnv`.
- `lux_ai/lux_gym/lux_env.py`: loads the official default config first, then merges custom `width`, `height`, and `loglevel`.

Therefore YAML configs can include:

```yaml
env_configuration:
  width: 16
  height: 16
  loglevel: 0
```

The same values can also be overridden from the command line:

```powershell
+env_configuration.width=32 +env_configuration.height=32
```

## Current Main Training

The current main training config is not 32x32. It is:

```text
conf/conv_teacher_finetune_16x16.yaml
```

Goals:

- Small-scale 100000-step validation.
- Save checkpoints every 10000 learner steps.
- Stabilize policy with teacher imitation / teacher KL.
- Use replays to inspect survival, expansion, research, and mining behavior.

## Extending To 24x24 / 32x32

Suggested order:

1. Continue fixing reward and behavior issues on 16x16.
2. Run a small 24x24 experiment and inspect expansion and fuel management.
3. Run a small 32x32 experiment and inspect whether workers spread effectively.
4. Use `conv_teacher_finetune_random_sizes.yaml` for mixed map-size training.

Related configs:

```text
conf/conv_teacher_finetune_24x24.yaml
conf/conv_teacher_finetune_32x32.yaml
conf/conv_teacher_finetune_random_sizes.yaml
```

## 32x32 Example Command

```powershell
$env:WANDB_MODE="offline"
.\.venv\Scripts\python.exe run_monobeast.py `
  --config-name conv_teacher_finetune_32x32 `
  actor_device=cuda:0 `
  learner_device=cuda:0 `
  disable_wandb=True
```

If VRAM is insufficient, reduce these first:

- `num_actors`
- `n_actor_envs`
- `batch_size`
- `unroll_length`
- `n_blocks`
- `hidden_dim`

## Replay Metrics

For large-map training, do not only check whether training completed. Inspect replays and record:

- Whether the agent survives to 360 turns.
- Whether it stays around one wood patch or expands to multiple resource areas.
- Whether research reaches 50 and 200.
- Whether city fuel covers night upkeep.
- Whether workers are congested or idle.
- Whether cities and workers remain stable under opponent pressure.

These metrics should decide whether a run is worth scaling to more training steps.
