#!/usr/bin/env python3
"""Train a LightGBM city-collapse risk scorer from strategy features."""

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

from train_city_loss_risk_scorer import FEATURES


def stable_split_key(file_value: str, episode_id: str, team: object, turn: object, val_fraction: float) -> bool:
    key = f"{file_value}:{episode_id}:{team}:{turn}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def load_frame(paths: list[Path], args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for path in paths:
        usecols = list(dict.fromkeys([*FEATURES, args.label_column, "file", "episode_id"]))
        frame = pd.read_csv(path, usecols=lambda col: col in usecols)
        if args.map_size:
            frame = frame[frame["map_size"].astype(int) == args.map_size]
        frames.append(frame)
    if not frames:
        raise ValueError("No input feature files.")
    data = pd.concat(frames, ignore_index=True)
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    for feature in FEATURES:
        if feature not in data.columns:
            data[feature] = 0.0
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data[args.label_column] = pd.to_numeric(data[args.label_column], errors="coerce").fillna(0.0)
    data["label"] = (data[args.label_column] >= args.loss_threshold).astype(int)
    data["is_val"] = [
        stable_split_key(str(file_value), str(episode_id), team, turn, args.val_fraction)
        for file_value, episode_id, team, turn in zip(
            data.get("file", ""),
            data.get("episode_id", ""),
            data.get("team", 0),
            data.get("turn", 0),
        )
    ]
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
    rows = []
    booster = model.booster_
    split_importance = booster.feature_importance(importance_type="split")
    gain_importance = booster.feature_importance(importance_type="gain")
    for feature, split, gain in zip(FEATURES, split_importance, gain_importance):
        rows.append({"feature": feature, "split": int(split), "gain": float(gain)})
    frame = pd.DataFrame(rows).sort_values("gain", ascending=False)
    frame.to_csv(output, index=False, encoding="utf-8")


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
        train_df["label"],
        eval_set=[(val_df[FEATURES], val_df["label"])],
        eval_metric="auc",
        callbacks=callbacks,
    )
    train_prob = model.predict_proba(train_df[FEATURES])[:, 1]
    val_prob = model.predict_proba(val_df[FEATURES])[:, 1]
    summary = {
        "features": [str(path) for path in args.features],
        "rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "label_column": args.label_column,
        "loss_threshold": args.loss_threshold,
        "decision_threshold": args.threshold,
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "train": compute_metrics(train_df["label"].to_numpy(), train_prob, args.threshold),
        "val": compute_metrics(val_df["label"].to_numpy(), val_prob, args.threshold),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": args.threshold,
            "label_column": args.label_column,
            "loss_threshold": args.loss_threshold,
        },
        args.output_dir / "risk_scorer_lgbm.joblib",
    )
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_feature_importance(model, args.output_dir / "feature_importance.csv")
    print(json.dumps(summary["val"], indent=2))
    print(f"model: {args.output_dir / 'risk_scorer_lgbm.joblib'}")
    print(f"importance: {args.output_dir / 'feature_importance.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM city-loss risk scorer.")
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/risk_scorer_lgbm_v1_top12_16"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--label-column", default="future_team_loss_10")
    parser.add_argument("--loss-threshold", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
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
