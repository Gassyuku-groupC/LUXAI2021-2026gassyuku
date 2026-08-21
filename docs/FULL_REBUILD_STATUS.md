# Lux Agent Full Rebuild Status

## Baseline

The current deployable behavior remains the immutable `best_agent` policy. The
external sidecar starts with an exactly zero logit projection. Four-map replay
verification at turn 80 reports identical checkpoint hashes, zero logit/value
difference, and 100% action agreement.

## Implemented

- Pooled-KV MHA spatial sidecar with detached Actor features and fixed 8x8 KV.
- Separate tile maps for `risk_big_loss_20` and `safe_expansion_success_40`.
- Zero-init additive delta gate. The safe-expansion score has priority over the
  risk penalty, and the original legality mask is applied after the delta.
- Map/phase rules: 12/16 disabled before turn 120 and active only at cycle
  turns 25-29; 24 player 0 disabled before turn 80; 32 always eligible.
- Existing hard runtime masking is disabled outside an explicit diagnostic
  mode. Production intervention is sidecar soft bias only.
- Replay-grouped scalar calibration already exists for every map and reaches
  precision >= 0.85 on raw calibration data.
- Frozen-Actor spatial sidecar training and grouped per-map calibration entry
  points, including five replay BC epochs by default.
- `appo_vtrace` learner mode: PPO clipped policy surrogate, V-trace critic
  target, no UPGO policy term, and logged clip fraction/approximate KL.
- Immutable reference policy loss uses `KL(pi_theta || pi_ref)` and the new
  `reference_policy_kl_cost` name, with backward compatibility.
- Outcome-dominant logarithmic city/unit reward with outcome weight >= 0.80.
- PFSP schedule sampler with hard-opponent, uncertainty, and night-loss
  weighting, plus a pool containing best, first, stage350, stage400 and a
  public external baseline.
- Progressive 8/16/24 ResNet configs and a compatible checkpoint migration
  script.
- Optional per-feature embedding dimensions for embedding observation spaces.
  The current best agent uses `FixedShapeContinuousObsV2`, so this option does
  not change or accelerate its continuous input encoder.

## Verified

- Component unit tests pass.
- Frozen base trainable parameters: 0; sidecar/gate trainable parameters: 62,852.
- GPU sidecar training smoke: two optimizer batches completed and checkpointed.
- GPU APPO smoke: one 12x12 game, 368 learner steps, clean exit.
- Four-map grouped spatial calibration smoke emits independent PR artifacts.

Smoke data is deliberately too small for model promotion. Its thresholds and
AP values are pipeline checks, not research results.

## Remaining Experiments

- Extract the full grouped shards, train five complete epochs, and calibrate on
  the held-out raw replay groups for all four maps.
- Update the APPO config with those spatial thresholds and run the full KL-APPO
  budget.
- Feed completed match statistics back into the PFSP pool and execute the
  sampled external-opponent schedule. The built-in TorchBeast environment is
  symmetric self-play; external-opponent PFSP remains an orchestration step.
- Run paired promotion series against best, first, stage350, stage400 and
  public baselines. No claim of outperforming `best_agent` is valid before
  these held-out results pass.

## Fallback Semantics

Lux 2021 `transfer` targets an adjacent unit, not a city. A legal hard-diagnostic
fallback is therefore: no-op while standing on a resource (mine), otherwise a
legal move into/toward a low-fuel friendly city to deposit cargo, otherwise the
next legal non-build-city action. Production mode does not hard-block actions.
