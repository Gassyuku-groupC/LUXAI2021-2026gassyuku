# Training And Diagnostics

This repository does not include replay datasets or processed training outputs. To reproduce the runtime scorer gate, prepare replay JSON files locally and run the diagnostic pipeline.

## 1. Prepare Data

Place Lux AI 2021 replay JSON files in a local directory. A typical layout is:

```text
dataset/
  raw/
    data/
      replay_001.json
      replay_002.json
```

Large replay files are intentionally excluded from this repository.

## 2. Build Strategy Labels

Create a row-level strategy label dataset from replay files:

```bash
python scripts/build_strategy_label_dataset.py \
  --replay-dir dataset/raw/data \
  --output dataset/processed/strategy_label_dataset.csv
```

Use `--help` to inspect the exact options supported by the local script version.

## 3. Validate Labels

Run the basic label consistency checks:

```bash
python scripts/validate_strategy_labels.py \
  --labels dataset/processed/strategy_label_dataset.csv \
  --output-md dataset/processed/strategy_label_validation.md
```

The labels are intended for diagnostics and counterfactual risk scoring, not as direct actor targets.

## 4. Train Risk Scorers

Train the LightGBM strategy scorers:

```bash
python scripts/train_strategy_label_scorers.py \
  --labels dataset/processed/strategy_label_dataset.csv \
  --output-dir outputs/diagnostic_layer/strategy_label_scorers
```

The runtime gate currently expects these scorer files:

```text
risk_big_loss_20_lgbm.joblib
error_failed_big_loss_lgbm.joblib
```

Copy the trained scorer files into:

```text
lux_ai/rl_agent/strategy_scorers/
```

## 5. Evaluate The Agent

Run local evaluation from the Lux AI project root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\evaluate_agent_generalization.ps1 `
  -CandidateAgent . `
  -OutputDir .\outputs\diagnostic_layer\gate_eval `
  -RandomSeedCount 2 `
  -EvaluationMapSizes 12,16,24 `
  -ReplayTimeoutSeconds 240 `
  -MaxReplayAttempts 1 `
  -ContinueOnReplayFailure
```

32x32 maps can be much slower and should be evaluated separately with a longer timeout.

## Notes

- The base neural actor is not directly fine-tuned by this gate pipeline.
- The gate is a runtime diagnostic layer that filters only selected high-risk actions.
- If scorer loading fails at runtime, the agent falls back to the base policy.
