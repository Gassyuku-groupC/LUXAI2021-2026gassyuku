# Diagnostic Layer Roadmap

This branch changes the improvement strategy from direct actor fine-tuning to
non-intrusive diagnosis first.

## Why actor fine-tuning failed

The current best agent is a coordinated policy. Small actor changes can shift
the state visitation distribution, where the base policy no longer has reliable
local behavior. In our experiments this appeared as:

- `v12/v12b`: more RL or stronger scale reward amplified side gap and caused p1
  collapse.
- `v3 BC`: risk-heavy reweighting reduced city loss but also reduced scale.
- `v4 BC`: safe-expansion reweighting still damaged the base policy and failed
  survival checks.

The conclusion is that replay features are useful, but direct full-actor updates
are too intrusive for now.

## Phase 1: Offline diagnostic models

Build independent models that only read replay/eval features.

1. City collapse risk scorer
   - Input: per-turn strategy/action features.
   - Label: `future_team_loss_10 > 0` or `>= 5`.
   - Output: `P_loss_10`.
   - First implementation: PyTorch tabular MLP, no new dependency.

2. Safe expansion scorer
   - Input: states with `bcity_actions > 0`.
   - Label: future no city loss plus high final scale/rank.
   - Output: safe expansion score.

## Phase 2: Automated diagnostics

For each evaluation run, produce:

- p0/p1 side gap and survival.
- phase hotspots: `040-079`, `080-119`, `120-159`.
- fuel-buffer curves and top replay deltas.
- risk scorer warnings before actual city loss.
- safe expansion opportunities missed by the agent.

## Phase 3: Non-intrusive control

Only after risk scoring is validated:

1. Log-only mode
   - Record actor action, risk score, final outcome.
2. Narrow safety gate
   - Only intercept extremely risky `bw`/`bcity` near night.
   - Do not touch movement, transfer, or resource micro.
3. Optional constrained learning
   - KL-constrained or head-only fine-tuning against the frozen best agent.

## Current rule

Do not promote BC/RL variants unless they beat the current safe baseline under
random-seed evaluation. The production-safe agent remains:

`outputs/auto_league_dagger_v10_shadow/best_agent`
