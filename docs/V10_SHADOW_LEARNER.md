# V10 Cumulative Shadow Learner

V10 keeps the stage 400 agent immutable as the champion while allowing a
separate shadow learner to accumulate useful updates.

## Schedule

- Train 256 games by default on balanced 12, 16, 24, and 32 maps.
- Save a candidate agent every 16 games.
- Run the complete two-sided 12/16/24/32 promotion evaluation every 64 games.
- A strict rejection does not reset the shadow learner.
- Replace the champion only when every existing promotion check passes.
- Roll back the shadow learner when any matchup has survival below 0.50 or
  night loss above 40, or when a majority of matchup groups has side city gap
  above 0.80.

The strict champion side-gap limit remains 0.35.

## Run

From the repository root in the VS Code PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\auto_train_shadow_v10.ps1
```

Resume at a completed game count with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\auto_train_shadow_v10.ps1 `
  -StartAtGames 64 -TotalGames 256
```

The active cumulative learner is stored in
`outputs/auto_league_dagger_v10_shadow/learner_agent`. The strict champion is
stored separately in `best_agent`. Every 16-game candidate remains in its
`game_stage_*` directory for later analysis.
