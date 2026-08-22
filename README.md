# Lux AI 2021 Actor-Sidecar Agent

This repository extends the open-source Lux AI 2021 first-place codebase with a replay-trained spatial risk sidecar and a KL-constrained APPO continuation stage. The current route starts from the first-place-era Actor weights, learns replay features with behavior cloning, and then performs low-learning-rate APPO + V-trace updates while a frozen `best_agent` supplies the reference policy.

The repository contains code and configuration only. Replays, model weights, scorer artifacts, Hydra outputs, and local agents are intentionally excluded by `.gitignore`.

## Current Architecture

```text
observation
  -> legacy 24-block ResNet Actor/Critic
  -> actor feature map (detached at the sidecar boundary)
  -> pooled-KV MHA spatial risk sidecar
  -> risk and safe-expansion maps
  -> zero-initialized additive logit-delta gate
  -> legal-action mask
  -> action selection
```

The training route is:

```text
expert replays
  -> seed/replay-grouped shards
  -> Actor + Sidecar behavior cloning
  -> low-LR APPO + V-trace
  -> frozen best_agent KL reference
  -> paired replay evaluation before promotion
```

Key properties:

- The legacy Actor parameter names remain checkpoint-compatible.
- Pooled key/value attention uses `AdaptiveAvgPool2d((8, 8))`, so query cost scales as `H*W*64` instead of full `H*W` self-attention.
- The final delta projection is zero-initialized; Step 0 policy logits match the base policy.
- Illegal actions are masked after adding the delta.
- Risk thresholds are calibrated independently for 12, 16, 24, and 32 maps.
- Small-map and early-game phase rules protect expansion tempo.
- APPO uses PPO clipping for policy optimization and V-trace/TD-lambda targets for asynchronous correction and critic learning.

See [METHODOLOGY.md](METHODOLOGY.md) for design details and [TRAINING.md](TRAINING.md) for reproducible commands.
See [ROLE.md](ROLE.md) for the optional role and city adapter, its Lux action
constraints, initial A/B evidence, and staged learning path.

## Repository Layout

```text
conf/
  conv_sidecar_bc_stage.yaml
  conv_sidecar_appo_vtrace.yaml
  league_pool_sidecar.json
  progressive_resnet_{8,16,24}.yaml

lux_ai/rl_agent/
  spatial_risk_sidecar.py
  learned_intervention_gate.py
  sidecar_agent_wrapper.py
  gate_policy.py
  role_assignment.py
  role_city_adapter.py

lux_ai/torchbeast/
  monobeast.py
  pfsp.py

scripts/
  replay generation, shard extraction, BC, spatial-risk calibration,
  Step-0 verification, PFSP sampling, and checkpoint migration
```

## Verified Training Status

The 200-game mixed-map KL-APPO smoke run completed on 2026-08-22:

- 200 completed games and 70,560 learner steps
- 12, 16, 24, and 32 map sampling enabled
- no non-finite loss or gradient after the masked-logit and AMP fixes
- throughput generally 25-30 SPS on the test machine
- Actor tensors changed: 176/177
- intervention-gate tensors changed: 8/8
- all final checkpoint tensors finite

The spatial risk sidecar remained unchanged during APPO because its risk outputs are deliberately detached before the intervention gate. It is learned in the replay/BC stage and treated as a fixed calibrated diagnostic during RL. The smoke run validates training mechanics, not competitive promotion; paired games against `best_agent`, first, stage350, and stage400 are still required.

The final CUDA IPC line printed during process shutdown is a worker cleanup warning. A run is considered complete only when the log contains `Learning finished` and the final checkpoint exists.

## Attribution

Based on [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) and its Lux AI 2021 first-place training stack. Original attribution and license terms remain applicable.
