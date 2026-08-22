# Methodology

## Objective

The project aims to reproduce the strength of `best_agent` and then improve it without breaking legacy checkpoint compatibility. The adopted method separates strategic policy learning from spatial risk diagnosis:

- the Actor remains the primary policy;
- a sidecar estimates tile-level future city-loss risk and safe expansion;
- a narrow additive gate changes only selected policy logits;
- low-learning-rate APPO updates are constrained by a frozen `best_agent` reference.

## Replay Splitting And Calibration

Replay samples are grouped by source replay and seed before train, validation, and calibration splitting. Frames from one replay must never be distributed across splits because adjacent Lux states are highly correlated.

Raw external replay data is the main independent calibration source. Newly generated deployed-agent replays represent the target policy distribution and are used for training and evaluation. Calibration is performed separately for 12, 16, 24, and 32 maps. The 32-map deployed set receives lower weight when it is small, with raw replay groups supplying broader coverage.

For each map size, calibration reports:

- replay/seed group count and frame count;
- precision-recall curve and average precision;
- the earliest threshold satisfying precision >= 0.85;
- the configured deployment threshold.

The current calibrated risk thresholds are:

| Map | Risk threshold |
| --- | ---: |
| 12 | 0.9582753841 |
| 16 | 0.9459641069 |
| 24 | 0.9389875956 |
| 32 | 0.8918934057 |

## Spatial Risk Sidecar

`SpatialRiskAttentionSidecar` is external to the legacy Actor state dictionary. It consumes `actor_features.detach()`, projects features to 64 channels, and forms full-resolution query tokens. Key/value tokens are generated with adaptive 8x8 pooling, followed by four-head cross-attention and spatial convolution. It outputs two `B x 2 x H x W` maps:

- future city-loss risk;
- safe-expansion probability.

The detached input prevents gradients from the sidecar training objective from modifying the legacy backbone. The fixed 64-token key/value sequence avoids quadratic full-board self-attention on 32x32 maps.

## Intervention Gate

The gate produces an additive delta for selected actions, currently `worker/BUILD_CITY` and `city_tile/BUILD_WORKER`. The final operation is:

```text
final_logits = legal_mask(base_logits + gate_delta)
```

The final projection starts with zero weights and bias, giving exact Step-0 equivalence. The safe-expansion signal has whitelist priority. Dynamic activation rules are:

- 12x12 and 16x16: disabled before turn 120; afterward active only when `turn % 40` is 25 through 29;
- 24x24: disabled for player 0 before turn 80;
- 32x32: active under its independently calibrated threshold.

The spatial risk logits are detached at the gate boundary during APPO. This is intentional: APPO can learn how strongly to intervene without corrupting the independently calibrated risk estimator. Sidecar weights are updated during replay supervision/BC, then fixed during KL-APPO.

## Behavior Cloning

The student starts from the historical first-place Actor checkpoint. Expert replay shards supervise policy actions and spatial-risk targets. Invalid expert targets are removed using the legal-action mask before cross-entropy. Training and validation are split by replay group rather than frame.

BC checkpoints contain the combined Actor, Sidecar, and Gate state. The number of epochs is selected by grouped validation convergence rather than fixed at five. Training stops or rolls back on non-finite loss, non-finite gradients, or worsening validation loss.

## KL-APPO + V-trace

The RL stage uses:

- PPO clipped surrogate policy loss;
- V-trace importance correction for asynchronous rollouts;
- TD-lambda critic targets;
- a frozen `best_agent` teacher with `KL(pi_student || pi_ref)`;
- terminal outcome weight 0.80 and logarithmic city/unit scale shaping;
- optional PFSP opponent sampling from best, historical, baseline, and external agents.

The default APPO learning rate is `1e-6` and the reference KL cost is `0.005`. Teacher BC is annealed from `0.10` to zero over 500 games. Mixed precision uses an initial gradient scale of 16 because the original `65536` scale overflowed the sum-reduced value loss.

## Promotion Criteria

A successful training run is not automatically a promoted agent. Promotion requires paired fixed-seed evaluation on both sides and all four map sizes against `best_agent`, first, stage350, and stage400. Report at least:

- win rate;
- final city-tile and unit margins;
- worst-night city loss;
- BUILD_CITY frequency;
- side-specific performance;
- timeout or invalid replay count.

The 200-game smoke run validates finite optimization and checkpoint updates only. It does not yet establish parity with or superiority over `best_agent`.

## Optional Role Adapter

The role and city adapter is an external, independently switchable policy
post-processor. It preserves the legacy Actor checkpoint and applies only
legal-action logit biases. The default configuration is disabled, which keeps
the reproduced `best_agent` path unchanged. See [ROLE.md](ROLE.md) for role
semantics, transfer constraints, learning stages, and current A/B evidence.
