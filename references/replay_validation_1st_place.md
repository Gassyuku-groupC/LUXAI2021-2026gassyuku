# Replay Validation Notes

This file records the replay validation workflow. The project needs to validate two kinds of replays:

1. Whether the original first-place agent runs correctly in the local environment.
2. Whether our locally trained agent can generate replay JSON files that work with the official visualizer.

## First-Place Agent Validation

The original first-place hall-of-fame agent is:

```text
internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/main.py
```

It comes from the base repository `IsaiahPressman/Kaggle_Lux_AI_2021` and is used as a teacher/reference agent.

Earlier local validation command:

```powershell
.\node_modules\.bin\lux-ai-2021.CMD `
  internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research\main.py `
  internal_testing\public_agents\working_title_bot_tong_hui_kang\main.py `
  --python ".\.venv\Scripts\python.exe" `
  --loglevel 2 `
  --memory 8000 `
  --maxtime 20000 `
  --width 12 `
  --height 12 `
  --storeLogs=true `
  --statefulReplay=true `
  --out replays\first_place_validation_12x12.json
```

That replay is no longer part of the main route. Its purpose was to verify that the original agent and the official Lux CLI could run locally.

## Current Agent Validation

Current main replay:

```text
replays/teacher_finetune_16x16_100000_vs_public_16x16_seed12345.json
```

Corresponding agent:

```text
local_agents/teacher_finetune_16x16_100000
local_agents/teacher_finetune_16x16_100000.zip
```

This replay verifies that:

- `100000_weights.pt` was successfully packaged into an agent.
- A full 16x16 match can be generated locally.
- The output JSON can be uploaded to the official visualizer.
- The result can be used as a baseline for future checkpoint and strategy comparisons.

## Evaluation Checklist

Future replays should not be judged by win/loss only. We should also record:

- Whether the agent survives to 360 turns.
- Whether city tile and city count grow in a healthy way.
- Whether workers spread across resources or stay stuck around one wood patch.
- Whether research reaches 50 and 200, enabling coal and uranium.
- Whether cities have enough fuel buffer before night.
- Whether units become congested, idle, or suddenly disappear.

These signals are more useful than a single match result when deciding how to train next.
