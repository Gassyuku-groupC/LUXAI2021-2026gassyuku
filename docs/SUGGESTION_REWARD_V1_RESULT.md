# Suggestion Reward V1 Result

## Branch

Git branch:

- `codex/suggestion-reward-v1`

Baseline agent:

- `outputs/auto_league_dagger_v10_shadow/best_agent`

This experiment does not modify the baseline actor. It trains an offline tabular suggestion reward model from replay-derived labels.

## Data

Input labels:

- `outputs/diagnostic_layer/suggestion_labels_v1/suggestion_labels.csv`

Rows:

- total: `2184`
- penalty rows: `1882`
- positive rows: `26`

Label semantics:

- `ignored_then_loss`: negative reward
- `ignored_without_loss`: zero reward
- `accepted_without_loss`: small positive reward
- `accepted_but_loss`: zero reward

## Model

Script:

- `scripts/train_suggestion_reward_lgbm.py`

Valid model:

- `outputs/diagnostic_layer/suggestion_reward_lgbm_v1b_from_best/suggestion_reward_lgbm.joblib`

The first run, `suggestion_reward_lgbm_v1_from_best`, included label/future leakage features and should not be used. The corrected run is `v1b`.

## V1B Validation

Penalty classifier:

- validation rows: `445`
- positive rate: `0.8607`
- AUC: `0.9797`
- average precision: `0.9962`
- accuracy: `0.9730`
- precision: `0.9843`
- recall: `0.9843`
- F1: `0.9843`

Reward regressor:

- validation MAE: `0.2283`
- validation RMSE: `0.3693`
- target mean: `-1.9838`
- prediction mean: `-1.9786`

Top penalty features:

- `cycle_turn`
- `ignored_suggestion`
- `team_city_tiles`
- `turn`
- `city_fuel_turns`
- `city_fuel`
- `city_upkeep`

## Interpretation

The model is good enough for a first diagnostic reward scorer. It should not directly control actions. The next safe use is:

- score replay/action logs;
- add an auxiliary reward or loss only during training;
- keep the production best agent unchanged until a non-invasive integration is validated.
