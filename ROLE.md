# Role and City Adapter

## Purpose

`RoleCityAdapter` is an optional policy adapter applied after Actor inference and
before action selection. It does not change the observation space, legacy Actor
parameter names, or base checkpoint. The adapter can therefore be enabled for
experiments and disabled to recover the reproduced `best_agent` policy.

The module is experimental and is disabled by default.

## Data Flow

```text
base Actor logits
  + per-turn unit roles and city specializations
  + configured or learned role biases
  -> legal-mask-respecting biased logits
  -> action selection
```

`RoleCityAdapter.update(...)` refreshes assignments and cooldown state once per
turn. `RoleCityAdapter.apply(...)` adds biases only to legal actions. An adapter
with `enabled: false` returns the original logits object unchanged.

## Assignments

Worker roles are `Harvester`, `Builder`, `Attacker`, and `Firefighter`. City
specializations are `FuelDepot`, `FuelStation`, `ResearchStation`,
`ManufacturingPoint`, and `SacrificialDecay`.

Assignments use a five-turn cooldown to avoid oscillation. Firefighter may
override cooldown only for a critical city that has not been classified as
`SacrificialDecay`. FuelDepot represents a combined transport hub rather than
the nearest fuel tile.

Lux 2021 transfer semantics are preserved: a worker can transfer resources only
to an adjacent allied unit. Firefighter transfer bias is added only when that
adjacent relay is no farther from the critical city. The adapter never assumes
that a worker can transfer fuel directly to a city tile.

## Configuration

The deployment default in `rl_agent_config.yaml` is:

```yaml
role_assignment:
  enabled: false
  dry_run_logging: false
  bias_enabled: false
  learnable_biases: false
  bias_params_path: role_city_bias_params.yaml
  annotate_summary: false
  cooldown_turns: 5
  firefighter_override_cooldown: true
```

Fixed deployment coefficients live in `role_city_bias_params.yaml`. A separate
role-enabled package can be generated with:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_checkpoint_agents.py `
  --checkpoint best_actor_sidecar_roles=outputs\best_actor_sidecar\best_actor_sidecar_zero_delta.pt `
  --disable-risk-gate `
  --preserve-runtime-config `
  --enable-role-adapter
```

## Learning Path

With `learnable_biases: true`, the 14 coefficients are represented by an
`nn.ParameterDict` and receive gradients through biased logits. Competitive
training should proceed in controlled stages:

1. Train Role Adapter parameters while Actor and Sidecar remain frozen.
2. Train Role Adapter and Sidecar while keeping the legacy Actor frozen.
3. Jointly fine-tune Actor, Sidecar, and Role Adapter with separate learning
   rates and a frozen `best_agent` KL reference.

Recommended initial learning-rate ranges are `1e-7` to `5e-7` for Actor,
`5e-7` to `2e-6` for Sidecar, and `5e-6` to `2e-5` for Role Adapter. Exposing
parameters is not sufficient by itself. The Role-only learner now transports a
compact signed action-to-parameter code with each rollout, reconstructs the
role Logit Delta in the learner, and saves the 14 parameters with optimizer and
checkpoint state. Actor, Sidecar, and Gate tensors remain frozen in this stage.

The Role-only repair configuration is `conf/conv_role_only_repair.yaml`. It uses
the reproduced best Actor plus zero-delta Sidecar as its student start, fixes
`best_agent` as the KL teacher, gives 12x12 a low policy weight as an anchor, and
up-weights turns 25-39 on 16x16 and 24x24. Learned values can be exported to the
runtime adapter YAML with `scripts/export_role_bias_checkpoint.py`.

## Bounded Multi-Map Runtime

Role assignment runs on all four map sizes. The deployment package uses full
learned role and city biases on 16x16. On 12x12, 24x24, and 32x32 it uses a
safety-only mode: Firefighter movement, adjacent-unit relay transfer, and the
Firefighter BUILD_CITY penalty remain active, while expansion, attacker, and
city-specialization biases are observe-only. Per-map bias scales and worker
budgets bound both strategy drift and runtime cost.

This replaces the earlier whole-map disable switch. The previously timing-out
24x24 seeds `314159265/p0` and `86753091/p1` completed under the 300-second
evaluation cap in safety-only mode. A 32x32 smoke with seed `20260826` also
completed under the same cap.

## Replay Overlay

Local replay generation writes the candidate player's actual cooldown-adjusted
assignments to a matching `*.roles.json` file. Each frame includes unit and city
roles, desired role, cooldown, assignment-change flag, reason, tile positions,
and whether the learned bias was active.

Open `tools/role_replay_viewer/index.html`, then select the converted stateful
replay and matching role sidecar. The offline viewer provides role colors,
unit/city and team filters, hover details, playback speed, and turn controls.
Pass `-DisableRoleTrace` to `generate_deployed_agent_replays.ps1` when role
sidecars are not needed.

## Historical A/B Evidence

The following early result used a package that did not preserve the base
agent's `Rot180` runtime augmentation. It is retained only as historical
evidence that the adapter executed; it must not be used for promotion.

The first paired fixed-seed evaluation against `best_agent` completed six games
on 12x12, 16x16, and 24x24. Both 32x32 games hit the existing 900-second Lux
evaluation timeout.

| Map | Record | Paired city margin | Paired unit margin |
| --- | ---: | ---: | ---: |
| 12x12 | 1-1 | +19 | +19 |
| 16x16 | 2-0 | +16 | +12 |
| 24x24 | 1-1 | 0 | -26 |

Across completed games the result was 4-2 with mean city margin `+5.83` and mean
unit margin `+0.83`. BUILD_CITY increased from the reproduced baseline's `88.5`
to `110.3` per game. The 24x24 games recorded worst-night city losses of 169 and
175, so this small result is evidence that the adapter is operational, not proof
that it is stronger than `best_agent`. Fresh-seed replication and night-loss
control are required before promotion.
