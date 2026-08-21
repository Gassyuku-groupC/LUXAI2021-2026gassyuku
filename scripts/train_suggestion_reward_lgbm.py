#!/usr/bin/env python3
"""Train a tabular suggestion reward model from offline suggestion labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURES = [
    "map_size",
    "turn",
    "cycle_turn",
    "turns_to_night",
    "is_night",
    "team",
    "eval_side",
    "unit_x",
    "unit_y",
    "unit_cargo_fuel",
    "suggested_city_x",
    "suggested_city_y",
    "city_fuel_turns",
    "city_fuel",
    "city_upkeep",
    "city_tiles",
    "team_city_tiles",
    "ignored_suggestion",
]

CATEGORICAL_FEATURES: list[str] = []


def stable_split_key(row: pd.Series, val_fraction: float) -> bool:
    key = f"{row.get('source_file', '')}:{row.get('team', '')}:{row.get('turn', '')}:{row.get('unit_id', '')}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def encode_categoricals(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Keep the first model intentionally deployable: no opponent ids, no
    # outcome labels, and no final-game leakage.
    return data, {}


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(args.labels)
    if args.map_size:
        data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == args.map_size]
    if args.max_turn:
        data = data[pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int) <= args.max_turn]
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if data.empty:
        raise ValueError("No suggestion labels loaded.")
    data, mappings = encode_categoricals(data)
    for col in [*FEATURES, *CATEGORICAL_FEATURES, "penalty_label", "reward_value"]:
        if col not in data.columns:
            data[col] = 0.0
    data[[*FEATURES, *CATEGORICAL_FEATURES, "penalty_label", "reward_value"]] = data[
        [*FEATURES, *CATEGORICAL_FEATURES, "penalty_label", "reward_value"]
    ].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data["is_val"] = data.apply(lambda row: stable_split_key(row, args.val_fraction), axis=1)
    return data, mappings


def classifier_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def regressor_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "target_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(pred)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
    }


def write_importance(model, features: list[str], output: Path) -> None:
    booster = model.booster_
    rows = []
    for feature, split, gain in zip(
        features,
        booster.feature_importance(importance_type="split"),
        booster.feature_importance(importance_type="gain"),
    ):
        rows.append({"feature": feature, "split": int(split), "gain": float(gain)})
    pd.DataFrame(rows).sort_values("gain", ascending=False).to_csv(output, index=False, encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    data, mappings = load_data(args)
    train_df = data[~data["is_val"]].copy()
    val_df = data[data["is_val"]].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty; adjust --val-fraction.")

    features = [*FEATURES, *CATEGORICAL_FEATURES]
    pos = max(int(train_df["penalty_label"].sum()), 1)
    neg = max(len(train_df) - pos, 1)
    scale_pos_weight = min(neg / pos, args.max_pos_weight)

    penalty_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    reward_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=args.log_period),
    ]
    penalty_model.fit(
        train_df[features],
        train_df["penalty_label"],
        eval_set=[(val_df[features], val_df["penalty_label"])],
        eval_metric="auc",
        callbacks=callbacks,
    )
    reward_model.fit(
        train_df[features],
        train_df["reward_value"],
        eval_set=[(val_df[features], val_df["reward_value"])],
        eval_metric="l2",
        callbacks=callbacks,
    )

    train_prob = penalty_model.predict_proba(train_df[features])[:, 1]
    val_prob = penalty_model.predict_proba(val_df[features])[:, 1]
    train_reward = reward_model.predict(train_df[features])
    val_reward = reward_model.predict(val_df[features])
    summary = {
        "baseline_agent_dir": str(args.baseline_agent_dir),
        "labels": str(args.labels),
        "rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "features": features,
        "categorical_mappings": mappings,
        "threshold": args.threshold,
        "scale_pos_weight": float(scale_pos_weight),
        "penalty_classifier": {
            "best_iteration": int(penalty_model.best_iteration_ or args.n_estimators),
            "train": classifier_metrics(train_df["penalty_label"].to_numpy(), train_prob, args.threshold),
            "val": classifier_metrics(val_df["penalty_label"].to_numpy(), val_prob, args.threshold),
        },
        "reward_regressor": {
            "best_iteration": int(reward_model.best_iteration_ or args.n_estimators),
            "train": regressor_metrics(train_df["reward_value"].to_numpy(), train_reward),
            "val": regressor_metrics(val_df["reward_value"].to_numpy(), val_reward),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "penalty_model": penalty_model,
            "reward_model": reward_model,
            "features": features,
            "categorical_mappings": mappings,
            "threshold": args.threshold,
            "baseline_agent_dir": str(args.baseline_agent_dir),
        },
        args.output_dir / "suggestion_reward_lgbm.joblib",
    )
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_importance(penalty_model, features, args.output_dir / "penalty_feature_importance.csv")
    write_importance(reward_model, features, args.output_dir / "reward_feature_importance.csv")
    print(json.dumps(summary["penalty_classifier"]["val"], indent=2))
    print(json.dumps(summary["reward_regressor"]["val"], indent=2))
    print(f"model: {args.output_dir / 'suggestion_reward_lgbm.joblib'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train suggestion reward scorer from offline labels.")
    parser.add_argument("--labels", type=Path, default=Path("outputs/diagnostic_layer/suggestion_labels_v1/suggestion_labels.csv"))
    parser.add_argument("--baseline-agent-dir", type=Path, default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/suggestion_reward_lgbm_v1_from_best"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--max-turn", type=int, default=240)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=8.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=17)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
