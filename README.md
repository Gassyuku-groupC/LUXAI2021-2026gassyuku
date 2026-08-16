# Lux AI 2021 Runtime Scorer Gate Agent

This repository contains a packaged Lux AI 2021 agent. The entry point is `main.py`.

## Package Contents

- `main.py`: Lux CLI/Kaggle entry point.
- `lux_ai/`: bundled agent source, Lux helper classes, model code, and runtime policy wrapper.
- `lux_ai/rl_agent/candidate_weights.pt`: base neural policy checkpoint.
- `lux_ai/rl_agent/strategy_scorers/`: lightweight risk scorers used by the runtime safety gate.
- `lux_ai/rl_agent/rl_agent_config.yaml`: runtime configuration.

Generated replay files, evaluation outputs, training shards, and datasets are intentionally excluded from this package.

## Runtime Idea

The agent keeps the original neural policy as the primary decision maker. A small diagnostic layer estimates late-game city-loss risk from scalar game-state features. The gate is intentionally conservative: it only intervenes on selected high-risk city-tile actions, currently focused on `BUILD_WORKER`, while leaving normal expansion and unit control to the base policy.

For 12x12, 16x16, and 24x24 maps the gate uses one shared timing profile. For 32x32 maps, intervention is delayed because large maps continue to benefit from expansion for longer.

## Dependencies

The packaged agent expects the Lux AI 2021 Python runtime plus the libraries used by the base neural agent. The risk scorer loader uses `joblib`; if scorer loading fails, the runtime gate is disabled and the base policy still runs.

Typical local environment:

```bash
pip install torch numpy pyyaml joblib lightgbm
```

## Local Match

From the Lux AI 2021 project root, this agent can be used as a player directory:

```bash
lux-ai-2021 path/to/this/agent path/to/opponent --python python --seed 12345 --width 16 --height 16
```

## Notes

- The safest tournament fallback remains the base agent without the gate.
- This gate package is intended for controlled experiments and solution demonstration.
- 32x32 local evaluations can be slow, especially against CPU-heavy opponents.
