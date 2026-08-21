#!/usr/bin/env python3
"""Train a learned action-level gate scorer.

This scorer is different from the earlier risk scorers. It does not only ask
"will a future loss happen?" Instead, it asks whether blocking the observed
candidate action is likely to be useful.

Positive labels are intentionally narrow:
- the observed candidate action is gateable, such as BUILD_WORKER or BUILD_CITY;
- the future contains a large loss / failed large-loss label;
- the final outcome is weak or the future scale collapses.

Negative labels include safe high-risk states and states where the same action
class does not lead to large loss. This keeps the learned gate from becoming a
blanket anti-expansion rule.
"""

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

from train_strategy_candidate_scorers_v2 import (  # noqa: E402
    ACTION_FEATURES,
    STATE_FEATURES,
    add_candidate_features,
)
from train_strategy_label_scorers import numeric_eval_side  # noqa: E402


SCORER_PROB_FEATURES = [
    "p_risk_city_loss_20",
    "p_risk_big_loss_20",
    "p_error_failed_big_loss",
    "p_safe_expansion_success_40",
    "p_success_stable_scale",
    "p_candidate_risk_city_loss_20",
    "p_candidate_risk_big_loss_20",
    "p_candidate_error_failed_big_loss",
    "p_candidate_safe_expansion_success_40",
    "p_candidate_success_stable_scale",
]

OUTCOME_FEATURES = [
    "final_city_tiles",
    "final_opponent_city_tiles",
    "future_city_tiles_gain_20",
    "future_city_tiles_gain_40",
]

FEATURES = STATE_FEATURES + ACTION_FEATURES + SCORER_PROB_FEATURES


def stable_group_is_val(file_value: str, episode_id: str, team: object, val_fraction: float) -> bool:
    key = f"{file_value}:{episode_id}:{team}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def numeric_series(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in data:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce").fillna(default)


def build_labels(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = data.copy()
    gateable = frame["candidate_action"].isin(args.gate_actions)
    big_loss = numeric_series(frame, "risk_big_loss_20").gt(0)
    failed_big_loss = numeric_series(frame, "error_failed_with_big_loss").gt(0)
    city_loss = numeric_series(frame, "risk_city_loss_20").gt(0)
    final_city = numeric_series(frame, "final_city_tiles")
    final_opp_city = numeric_series(frame, "final_opponent_city_tiles")
    future_gain_40 = numeric_series(frame, "future_city_tiles_gain_40")
    final_weak = final_city.lt(final_opp_city + args.min_final_margin)
    scale_bad = future_gain_40.le(args.max_bad_future_gain)
    safe_outcome = (
        ~city_loss
        | numeric_series(frame, "success_stable_scale").gt(0)
        | numeric_series(frame, "expansion_success_40").gt(0)
        | final_city.ge(final_opp_city + args.safe_final_margin)
    )

    positive = gateable & (failed_big_loss | (big_loss & (final_weak | scale_bad)))
    negative = (~gateable) | safe_outcome
    frame["action_gate_intervene"] = positive.astype(int)
    frame["action_gate_trainable"] = (positive | negative).astype(int)
    frame["action_gate_reason"] = np.where(
        positive,
        "intervene_failed_or_bad_big_loss",
        np.where(gateable, "keep_safe_or_unproven", "non_gateable_action"),
    )
    return frame


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.input_csv, low_memory=False)
    if args.map_size:
        data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == args.map_size].copy()
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if data.empty:
        raise ValueError("No rows loaded.")

    data["eval_side_numeric"] = numeric_eval_side(data.get("eval_side", pd.Series([""] * len(data))))
    data = add_candidate_features(data)
    data = build_labels(data, args)
    data = data[data["action_gate_trainable"].gt(0)].copy()
    if data.empty:
        raise ValueError("No trainable action-gate rows after labeling.")

    for column in FEATURES + OUTCOME_FEATURES + ["sample_weight", "file", "episode_id", "team"]:
        if column not in data:
            data[column] = 0.0
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data["sample_weight"] = pd.to_numeric(data["sample_weight"], errors="coerce").fillna(1.0).clip(lower=0.1, upper=10.0)
    data["sample_weight"] = data["sample_weight"] * np.where(data["action_gate_intervene"].gt(0), args.positive_weight, 1.0)
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
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        item = classifier_metrics(y_true, prob, threshold)
        item["threshold"] = threshold
        rows.append(item)
    return rows


def action_breakdown(data: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict:
    frame = data[["candidate_action", "action_gate_intervene", "action_gate_reason"]].copy()
    frame["prob"] = prob
    frame["alert"] = frame["prob"] >= threshold
    rows = {}
    for action, group in frame.groupby("candidate_action"):
        rows[str(action)] = {
            "rows": int(len(group)),
            "positive_rate": float(group["action_gate_intervene"].mean()),
            "mean_prob": float(group["prob"].mean()),
            "alert_rate": float(group["alert"].mean()),
            "reasons": group["action_gate_reason"].value_counts().to_dict(),
        }
    return rows


def write_importance(model: lgb.LGBMClassifier, output: Path) -> None:
    booster = model.booster_
    rows = []
    for feature, split, gain in zip(
        FEATURES,
        booster.feature_importance(importance_type="split"),
        booster.feature_importance(importance_type="gain"),
    ):
        rows.append({"feature": feature, "split": int(split), "gain": float(gain)})
    pd.DataFrame(rows).sort_values("gain", ascending=False).to_csv(output, index=False, encoding="utf-8")


def write_markdown(path: Path, summary: dict) -> None:
    val = summary["val"]
    lines = [
        "# Action Gate Scorer",
        "",
        f"- input: `{summary['input_csv']}`",
        f"- rows: {summary['rows']}",
        f"- map_size: {summary['map_size']}",
        f"- gate_actions: `{', '.join(summary['gate_actions'])}`",
        "",
        "| split | pos rate | AUC | AP | precision | recall | F1 | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| validation | {val['positive_rate']:.3f} | {val['auc']:.3f} | "
            f"{val['average_precision']:.3f} | {val['precision']:.3f} | "
            f"{val['recall']:.3f} | {val['f1']:.3f} | {summary['threshold']:.2f} |"
        ),
        "",
        "## Validation Thresholds",
        "",
        "| threshold | precision | recall | F1 | alerts |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["val_threshold_table"]:
        alerts = item["tp"] + item["fp"]
        lines.append(
            f"| {item['threshold']:.2f} | {item['precision']:.3f} | "
            f"{item['recall']:.3f} | {item['f1']:.3f} | {alerts} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned action-level runtime gate scorer.")
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/action_gate_scorer_v1"))
    parser.add_argument("--map-size", type=int, default=0, help="0 means all map sizes.")
    parser.add_argument("--gate-actions", nargs="+", default=["bw", "bcity"])
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--min-final-margin", type=float, default=0.0)
    parser.add_argument("--safe-final-margin", type=float, default=5.0)
    parser.add_argument("--max-bad-future-gain", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    data = load_data(args)
    train_df = data[~data["is_val"]].copy()
    val_df = data[data["is_val"]].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty.")
    if train_df["action_gate_intervene"].nunique() < 2 or val_df["action_gate_intervene"].nunique() < 2:
        raise ValueError("action_gate_intervene has fewer than two classes in train or validation.")
    pos = max(int(train_df["action_gate_intervene"].sum()), 1)
    neg = max(len(train_df) - pos, 1)
    scale_pos_weight = min(neg / pos, 20.0)

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
        train_df["action_gate_intervene"],
        sample_weight=train_df["sample_weight"],
        eval_set=[(val_df[FEATURES], val_df["action_gate_intervene"])],
        eval_sample_weight=[val_df["sample_weight"]],
        eval_metric="auc",
        callbacks=callbacks,
    )

    train_prob = model.predict_proba(train_df[FEATURES])[:, 1]
    val_prob = model.predict_proba(val_df[FEATURES])[:, 1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": args.threshold,
            "label_column": "action_gate_intervene",
            "gate_actions": args.gate_actions,
            "map_size": args.map_size,
            "label_policy": {
                "min_final_margin": args.min_final_margin,
                "safe_final_margin": args.safe_final_margin,
                "max_bad_future_gain": args.max_bad_future_gain,
            },
        },
        args.output_dir / "action_gate_scorer_lgbm.joblib",
    )
    write_importance(model, args.output_dir / "feature_importance.csv")

    summary = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "map_size": args.map_size,
        "rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "positive_rows": int(data["action_gate_intervene"].sum()),
        "threshold": args.threshold,
        "gate_actions": args.gate_actions,
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "features": FEATURES,
        "train": classifier_metrics(train_df["action_gate_intervene"].to_numpy(), train_prob, args.threshold),
        "val": classifier_metrics(val_df["action_gate_intervene"].to_numpy(), val_prob, args.threshold),
        "val_threshold_table": threshold_table(val_df["action_gate_intervene"].to_numpy(), val_prob),
        "val_by_candidate_action": action_breakdown(val_df, val_prob, args.threshold),
        "candidate_action_counts": data["candidate_action"].value_counts().to_dict(),
        "reason_counts": data["action_gate_reason"].value_counts().to_dict(),
        "model_path": str(args.output_dir / "action_gate_scorer_lgbm.joblib"),
    }
    (args.output_dir / "action_gate_scorer_training_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "action_gate_scorer_training_report.md", summary)
    print(json.dumps(summary["val"], indent=2, ensure_ascii=False))
    print(f"model: {args.output_dir / 'action_gate_scorer_lgbm.joblib'}")
    print(f"report: {args.output_dir / 'action_gate_scorer_training_report.json'}")


if __name__ == "__main__":
    main()
