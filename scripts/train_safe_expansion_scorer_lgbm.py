#!/usr/bin/env python3
"""Train a LightGBM scorer for high-quality safe city expansion opportunities."""

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


EXPANSION_FEATURES = [
    "map_size",
    "turn",
    "turns_remaining",
    "cycle_turn",
    "pre_night",
    "is_night",
    "turns_to_night",
    "team",
    "city_tiles",
    "cities",
    "largest_city_size",
    "mean_city_size",
    "resource_near_cities",
    "isolated_cities_r3",
    "workers",
    "worker_citytile_ratio",
    "unit_cap_margin",
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
]


def stable_split_key(file_value: str, episode_id: str, team: object, turn: object, val_fraction: float) -> bool:
    key = f"{file_value}:{episode_id}:{team}:{turn}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def load_frame(paths: list[Path], args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    usecols = list(dict.fromkeys([
        *EXPANSION_FEATURES,
        "file",
        "episode_id",
        "bcity_actions",
        "future_team_loss_10",
        "final_city_tiles",
        "city_tiles_delta_next",
        "rank",
    ]))
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda col: col in usecols)
        if args.map_size:
            frame = frame[frame["map_size"].astype(int) == args.map_size]
        frames.append(frame)
    if not frames:
        raise ValueError("No input feature files.")
    data = pd.concat(frames, ignore_index=True)
    data = data[pd.to_numeric(data["bcity_actions"], errors="coerce").fillna(0.0) > 0].copy()
    if args.min_turn:
        data = data[pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int) >= args.min_turn]
    if args.max_turn:
        data = data[pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int) <= args.max_turn]
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if data.empty:
        raise ValueError("No bcity expansion rows loaded.")
    for feature in EXPANSION_FEATURES:
        if feature not in data.columns:
            data[feature] = 0.0
    numeric_cols = list(dict.fromkeys([
        *EXPANSION_FEATURES,
        "future_team_loss_10",
        "final_city_tiles",
        "city_tiles_delta_next",
        "bcity_actions",
        "rank",
    ]))
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if args.final_city_quantile > 0:
        final_threshold = float(data["final_city_tiles"].quantile(args.final_city_quantile))
    else:
        final_threshold = args.final_city_threshold
    final_threshold = max(final_threshold, args.final_city_threshold)
    data["label"] = (
        (data["future_team_loss_10"] <= args.max_future_loss)
        & (data["rank"] <= args.max_rank)
        & (data["final_city_tiles"] >= final_threshold)
        & (data["city_tiles_delta_next"] > 0)
    ).astype(int)
    data["is_val"] = [
        stable_split_key(str(file_value), str(episode_id), team, turn, args.val_fraction)
        for file_value, episode_id, team, turn in zip(
            data.get("file", ""),
            data.get("episode_id", ""),
            data.get("team", 0),
            data.get("turn", 0),
        )
    ]
    data.attrs["final_city_threshold_used"] = final_threshold
    return data


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
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


def write_feature_importance(model: lgb.LGBMClassifier, output: Path) -> None:
    booster = model.booster_
    rows = []
    for feature, split, gain in zip(
        EXPANSION_FEATURES,
        booster.feature_importance(importance_type="split"),
        booster.feature_importance(importance_type="gain"),
    ):
        rows.append({"feature": feature, "split": int(split), "gain": float(gain)})
    pd.DataFrame(rows).sort_values("gain", ascending=False).to_csv(output, index=False, encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    data = load_frame(args.features, args)
    train_df = data[~data["is_val"]]
    val_df = data[data["is_val"]]
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty; adjust --val-fraction.")
    pos = max(int(train_df["label"].sum()), 1)
    neg = max(len(train_df) - pos, 1)
    scale_pos_weight = min(neg / pos, args.max_pos_weight)
    model = lgb.LGBMClassifier(
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
    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=args.log_period),
    ]
    model.fit(
        train_df[EXPANSION_FEATURES],
        train_df["label"],
        eval_set=[(val_df[EXPANSION_FEATURES], val_df["label"])],
        eval_metric="auc",
        callbacks=callbacks,
    )
    train_prob = model.predict_proba(train_df[EXPANSION_FEATURES])[:, 1]
    val_prob = model.predict_proba(val_df[EXPANSION_FEATURES])[:, 1]
    summary = {
        "features": [str(path) for path in args.features],
        "rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "positive_rows": int(data["label"].sum()),
        "decision_threshold": args.threshold,
        "final_city_threshold": float(data.attrs["final_city_threshold_used"]),
        "max_future_loss": args.max_future_loss,
        "max_rank": args.max_rank,
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "train": compute_metrics(train_df["label"].to_numpy(), train_prob, args.threshold),
        "val": compute_metrics(val_df["label"].to_numpy(), val_prob, args.threshold),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": EXPANSION_FEATURES,
            "threshold": args.threshold,
            "final_city_threshold": float(data.attrs["final_city_threshold_used"]),
            "max_future_loss": args.max_future_loss,
            "max_rank": args.max_rank,
        },
        args.output_dir / "safe_expansion_scorer_lgbm.joblib",
    )
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_feature_importance(model, args.output_dir / "feature_importance.csv")
    print(json.dumps(summary["val"], indent=2))
    print(f"model: {args.output_dir / 'safe_expansion_scorer_lgbm.joblib'}")
    print(f"importance: {args.output_dir / 'feature_importance.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM safe-expansion scorer.")
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/safe_expansion_lgbm_v1_top12_16"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-turn", type=int, default=0)
    parser.add_argument("--max-turn", type=int, default=0)
    parser.add_argument("--max-future-loss", type=float, default=0.0)
    parser.add_argument("--max-rank", type=float, default=1.0)
    parser.add_argument("--final-city-threshold", type=float, default=0.0)
    parser.add_argument("--final-city-quantile", type=float, default=0.80)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=8.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=7)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
