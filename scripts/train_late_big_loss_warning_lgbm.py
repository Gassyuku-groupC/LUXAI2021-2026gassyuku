#!/usr/bin/env python3
"""Train a LightGBM scorer for late-game big city loss warnings."""

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
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURES = [
    "map_size",
    "turn",
    "turns_remaining",
    "night_cycle",
    "cycle_turn",
    "pre_night",
    "is_night",
    "turns_to_night",
    "team",
    "eval_side",
    "city_tiles",
    "cities",
    "largest_city_size",
    "mean_city_size",
    "resource_near_cities",
    "isolated_cities_r3",
    "units",
    "workers",
    "carts",
    "unit_cap_margin",
    "worker_citytile_ratio",
    "research",
    "fuel",
    "upkeep",
    "fuel_turns_total",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "median_city_fuel_turns",
    "mean_city_fuel_turns",
    "low_fuel_city_lt3",
    "low_fuel_city_lt5",
    "low_fuel_city_lt10",
    "unit_cargo_fuel",
    "wood_remaining",
    "coal_remaining",
    "uranium_remaining",
    "action_count",
    "move_actions",
    "transfer_actions",
    "research_actions",
    "bw_actions",
    "bc_actions",
    "bcity_actions",
    "bcity_isolated_actions",
    "bcity_adjacent_actions",
    "bcity_resource_near_actions",
    "bcity_adjacent_low_fuel_lt5_actions",
    "bw_low_fuel_lt3_actions",
    "bw_low_fuel_lt5_actions",
    "bw_low_fuel_lt10_actions",
    "city_tiles_delta_10",
    "city_tiles_growth_10",
    "workers_delta_10",
    "upkeep_delta_10",
    "upkeep_growth_10",
    "fuel_delta_10",
    "fuel_turns_total_delta_10",
    "fuel_turns_total_drop_10",
    "p25_city_fuel_turns_delta_10",
    "p25_city_fuel_turns_drop_10",
    "min_city_fuel_turns_delta_10",
    "research_delta_10",
    "research_growth_10",
    "city_tiles_delta_next",
    "units_delta_next",
    "research_delta_next",
]


def stable_group_is_val(file_value: str, episode_id: str, team: object, val_fraction: float) -> bool:
    key = f"{file_value}:{episode_id}:{team}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.labels)
    if args.map_size:
        data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == args.map_size]
    if args.start_turn:
        data = data[pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int) >= args.start_turn]
    if args.max_turn:
        data = data[pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int) <= args.max_turn]
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if data.empty:
        raise ValueError("No late big-loss labels loaded.")

    for col in [*FEATURES, args.label_column, args.weight_column, "file", "episode_id"]:
        if col not in data.columns:
            data[col] = 0.0
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data[args.label_column] = pd.to_numeric(data[args.label_column], errors="coerce").fillna(0).astype(int)
    data[args.weight_column] = pd.to_numeric(data[args.weight_column], errors="coerce").fillna(1.0).clip(lower=0.1)
    data["is_val"] = [
        stable_group_is_val(str(file_value), str(episode_id), team, args.val_fraction)
        for file_value, episode_id, team in zip(data["file"], data["episode_id"], data["team"])
    ]
    return data


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


def threshold_table(y_true: np.ndarray, prob: np.ndarray) -> list[dict]:
    rows = []
    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        item = classifier_metrics(y_true, prob, threshold)
        item["threshold"] = threshold
        rows.append(item)
    return rows


def write_importance(model: lgb.LGBMClassifier, features: list[str], output: Path) -> None:
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
    data = load_data(args)
    train_df = data[~data["is_val"]].copy()
    val_df = data[data["is_val"]].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty; adjust --val-fraction.")

    pos = max(int(train_df[args.label_column].sum()), 1)
    neg = max(len(train_df) - pos, 1)
    scale_pos_weight = min(neg / pos, args.max_pos_weight)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
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
    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=args.log_period),
    ]
    model.fit(
        train_df[FEATURES],
        train_df[args.label_column],
        sample_weight=train_df[args.weight_column],
        eval_set=[(val_df[FEATURES], val_df[args.label_column])],
        eval_sample_weight=[val_df[args.weight_column]],
        eval_metric="auc",
        callbacks=callbacks,
    )

    train_prob = model.predict_proba(train_df[FEATURES])[:, 1]
    val_prob = model.predict_proba(val_df[FEATURES])[:, 1]
    summary = {
        "labels": str(args.labels),
        "rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "features": FEATURES,
        "label_column": args.label_column,
        "weight_column": args.weight_column,
        "threshold": args.threshold,
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "train": classifier_metrics(train_df[args.label_column].to_numpy(), train_prob, args.threshold),
        "val": classifier_metrics(val_df[args.label_column].to_numpy(), val_prob, args.threshold),
        "val_threshold_table": threshold_table(val_df[args.label_column].to_numpy(), val_prob),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": args.threshold,
            "label_column": args.label_column,
            "weight_column": args.weight_column,
            "start_turn": args.start_turn,
            "horizon": args.horizon,
            "big_loss_threshold": args.big_loss_threshold,
        },
        args.output_dir / "late_big_loss_warning_lgbm.joblib",
    )
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_importance(model, FEATURES, args.output_dir / "feature_importance.csv")
    print(json.dumps(summary["val"], indent=2))
    print(f"model: {args.output_dir / 'late_big_loss_warning_lgbm.joblib'}")
    print(f"importance: {args.output_dir / 'feature_importance.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train late big-loss warning LightGBM scorer.")
    parser.add_argument("--labels", type=Path, default=Path("outputs/diagnostic_layer/late_big_loss_warning_v1_public_v2_16/late_big_loss_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/late_big_loss_warning_lgbm_v1_public_v2_16"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--start-turn", type=int, default=160)
    parser.add_argument("--max-turn", type=int, default=360)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--big-loss-threshold", type=int, default=10)
    parser.add_argument("--label-column", default="late_big_loss_warning")
    parser.add_argument("--weight-column", default="late_big_loss_weight")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=60)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=12.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=29)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
