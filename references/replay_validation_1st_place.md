# 1st Place Replay Validation

Validated the final 1st place hall-of-fame agent by running one official Lux AI
2021 CLI match.

## Match

- Player 0: `internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/main.py`
- Player 1: `internal_testing/public_agents/working_title_bot_tong_hui_kang/main.py`
- Map: 12x12 random
- Seed: `220343070`
- Replay: `replays/first_place_validation_12x12.json`

## Result

```text
rank 1: agent 0
rank 2: agent 1
stateful turns: 192
command turns: 191
```

## Command

```powershell
$env:PATH="C:\Users\YE ZIHAN\.venvs\lux-ai-2021\Scripts;C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:PATH"
.\node_modules\.bin\lux-ai-2021.CMD internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research\main.py internal_testing\public_agents\working_title_bot_tong_hui_kang\main.py --python "C:\Users\YE ZIHAN\.venvs\lux-ai-2021\Scripts\python.exe" --loglevel 2 --memory 8000 --maxtime 20000 --width 12 --height 12 --storeLogs=true --statefulReplay=true --out replays\first_place_validation_12x12.json
```
