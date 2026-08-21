# Solution Overview

## Summary

This agent is based on a strong reinforcement-learning Lux AI 2021 policy, packaged with an additional runtime diagnostic safety layer. The main design goal is to reduce catastrophic city loss without directly overwriting the base policy's learned micro-control.

Earlier experiments showed that direct behavior cloning or actor fine-tuning can easily damage the original policy. Small policy changes shift the state visitation distribution, and the model may lose useful worker/city coordination learned by the base agent. For that reason, this version keeps the neural actor unchanged and adds a non-intrusive risk gate at inference time.

## Base Agent

The base policy is a neural RL agent with collision handling and action filtering. It remains responsible for normal gameplay:

- worker movement
- city expansion
- research progression
- transfer and collision-sensitive decisions
- most city-tile production choices

The packaged checkpoint is stored at:

```text
lux_ai/rl_agent/candidate_weights.pt
```

## Diagnostic Risk Layer

The runtime gate uses small offline-trained risk scorers stored in:

```text
lux_ai/rl_agent/strategy_scorers/
```

The scorers estimate large future city-loss risk from game-state features such as:

- map size
- turn and day/night cycle
- current city count and city tiles
- worker/city-tile ratio
- fuel, upkeep, and city fuel buffer statistics
- low-fuel city counts
- resource availability

The labels used to train the diagnostic layer were mined from replay data. The important target is not simply "avoid any loss", because the official ranking is determined by final city tiles and units. Instead, the scorer focuses on dangerous situations where city loss is large enough to threaten the final result.

## Runtime Gate

The gate is conservative. The base neural policy first proposes actions normally. The scorer is then evaluated once per turn. If risk is above threshold, only selected high-risk city-tile actions are filtered.

Current action scope:

```text
BUILD_WORKER
```

The gate does not broadly block city building. This preserves the base policy's expansion behavior and avoids turning the agent into an overly defensive strategy.

## Map-Size Handling

The gate uses a shared timing profile on 12x12, 16x16, and 24x24 maps. For 32x32 maps, intervention is delayed because expansion remains valuable for longer and premature safety gating can hurt scale.

Important configuration values are in:

```text
lux_ai/rl_agent/rl_agent_config.yaml
```

## Engineering Notes

The gate is designed as a fallback-safe wrapper. If the risk scorer dependencies or model files cannot be loaded, the runtime scorer gate is disabled and the base policy continues to run.

The package also limits CPU threading for more stable evaluation:

```text
runtime_torch_num_threads: 1
```

This helps avoid local timeout cascades when multiple Python agents are launched by the Lux CLI.

## Validation Status

The gate version has been validated as a packaged agent on small smoke tests. In current testing, it does not change results on some stable 12x12 and 16x16 games against the reference group agent. Further full-map validation is still required before using this branch as the primary tournament submission.

For high-stakes matches, the base agent remains the conservative fallback unless the gate branch shows a clear same-seed improvement.
