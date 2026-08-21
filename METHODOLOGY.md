# Methodology And Architecture

## Methodology

Our method starts from a strong reinforcement-learning agent and adds a non-intrusive runtime safety layer. The base neural policy remains the main decision maker. Instead of directly fine-tuning the actor, we mine replay data to train lightweight diagnostic models that estimate whether the current state is likely to lead to a large future city loss.

The motivation is that direct actor fine-tuning can easily damage a strong Lux AI policy. A small change in one worker or city-tile action can shift the future state distribution, and the policy may enter positions that the original agent rarely visited. In our experiments, behavior cloning and reward-only fine-tuning often reduced city loss in one dimension but also harmed expansion, unit count, or side stability. Therefore, we use a separate diagnostic layer as an external safety mechanism.

The runtime gate follows three principles:

1. Preserve the base policy whenever risk is low.
2. Intervene only in high-risk states predicted by the diagnostic scorer.
3. Restrict the intervention scope to selected dangerous actions, currently focused on city-tile `BUILD_WORKER`.

This makes the method different from a hand-written rule agent. The rule does not decide the full strategy. It only acts as a narrow safety filter around a learned RL policy.

## Replay-Based Risk Learning

The diagnostic layer is trained from replay-derived labels. For every replay state, we extract scalar features such as:

- map size and turn
- day/night cycle
- city tile count and unit count
- worker-to-city ratio
- fuel, upkeep, and fuel buffer statistics
- low-fuel city counts
- resource availability

The main labels describe future risk:

- `risk_big_loss_20`: whether a large city loss occurs within a future window
- `error_failed_big_loss`: whether the state is associated with both large loss and a bad final result

These labels allow the gate to focus on harmful city loss rather than treating every city loss as equally bad. This is important because Lux AI 2021 is decided by final city tiles and units, not directly by intermediate city loss.

## Runtime Architecture

```mermaid
flowchart TD
    A["Lux AI Observation<br/>game updates, map, units, city tiles, research"] --> B["Environment Wrappers<br/>FixedShapeContinuousObsV2<br/>pad board to 32x32"]
    B --> C["Input Features<br/>spatial tensor: C x 32 x 32<br/>+ legal action masks"]

    C --> D["Conv Embedding Input Layer<br/>embedding_dim=32<br/>hidden_dim=128"]
    D --> E["Residual CNN Trunk x24<br/>conv_model<br/>5x5 kernels"]

    subgraph RB["One Residual Block"]
        RB1["5x5 Conv2d"] --> RB2["LeakyReLU"]
        RB2 --> RB3["5x5 Conv2d"]
        RB3 --> RB4["SE-Layer<br/>Squeeze-and-Excitation"]
        RB4 --> RB5["Skip Connection<br/>x + residual"]
        RB5 --> RB6["LeakyReLU"]
    end

    E --> F["Shared Spatial Feature Map<br/>128 x 32 x 32"]
    F --> G["Actor Head Base<br/>1x1 Conv + SpectralNorm + ReLU"]
    G --> H["DictActor Policy Heads<br/>1x1 Conv per action space"]
    H --> I["Policy Logits<br/>worker and city-tile actions<br/>masked by legal actions"]

    F --> J["Value Head Base<br/>1x1 Conv + SpectralNorm + ReLU"]
    J --> K["Baseline Head<br/>masked average pooling<br/>Linear -> value"]

    I --> L["Action Selection<br/>arg-sort logits, no sampling"]
    L --> M["Collision / Legality Resolver"]

    C --> N["Scalar Strategy Feature Extractor<br/>turn, map size, day/night,<br/>city tiles, units, fuel/upkeep,<br/>p25 fuel buffer, low-fuel cities"]
    N --> O["LightGBM Risk Scorers<br/>all-map tabular models"]
    O --> P["Risk Scores<br/>risk_big_loss_20<br/>error_failed_big_loss"]

    M --> Q["Runtime Scorer Gate"]
    P --> Q
    Q --> R["Final Lux Actions"]
    R --> S["Lux Engine"]
```

## Neural Policy

The base policy is a ResNet-style actor-critic model trained with reinforcement learning. The packaged configuration uses:

```text
observation space: FixedShapeContinuousObsV2
model_arch:       conv_model
hidden_dim:       128
embedding_dim:    32
n_blocks:         24 residual blocks
kernel_size:      5
activation:       LeakyReLU
residual module:  Conv2d -> LeakyReLU -> Conv2d -> SE-Layer -> skip connection -> LeakyReLU
policy heads:     1x1 Conv actor heads for worker/city-tile action logits
value head:       1x1 Conv base + masked average pooling + linear baseline
runtime aug:      Rot180 test-time augmentation on small/medium maps
```

The neural network keeps full two-dimensional board structure throughout the convolutional trunk. This is important in Lux AI because local topology matters: city adjacency, worker position, nearby resources, opponent pressure, and collision constraints all depend on spatial layout.

The actor head outputs logits for each action space and board location. Legal-action masks are applied before action selection, and the resolver then handles collision and preference ordering. The value head is used by the RL training pipeline as the baseline estimator.

## Gate Logic

The gate does not replace this neural policy. If the scorer files are missing or fail to load, the system falls back to the base policy.

```mermaid
flowchart TD
    A["Actor proposes action<br/>example: BUILD_WORKER"] --> B{"Gate active?<br/>map / turn / night timing"}
    B -- "no" --> Z["keep actor action"]
    B -- "yes" --> C{"Risk scorer high?<br/>risk_big_loss_20 >= threshold<br/>or error_failed_big_loss >= threshold"}
    C -- "no" --> Z
    C -- "yes" --> D{"Action in gate target list?<br/>currently BUILD_WORKER"}
    D -- "no" --> Z
    D -- "yes" --> E["mark action unsafe<br/>remove from final candidates"]
    E --> F["resolver selects next legal action"]
```

At inference time, the scorer is evaluated once per turn. If predicted risk is below threshold, the base policy action is used unchanged. If risk is high, the gate marks selected dangerous actions as illegal before final action resolution.

Current intervention target:

```text
BUILD_WORKER
```

The gate is map-size aware. For 12x12, 16x16, and 24x24 maps it uses a shared timing profile. For 32x32 maps, intervention is delayed because large maps benefit from expansion for longer.

## Future Extension

The current gate threshold is manually configured. A natural next step is to learn the gate decision itself:

- train an action-level intervention classifier from replay outcomes
- optimize thresholds on held-out seeds
- use offline contextual bandit learning to decide whether blocking an action improves final win/city outcome
- keep a KL/behavior anchor to the base policy during any further RL adaptation

This would preserve the current architecture while reducing manual threshold tuning.
