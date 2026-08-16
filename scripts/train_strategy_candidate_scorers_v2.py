#!/usr/bin/env python3
"""Train pre-action-ish candidate-action strategy scorers.

V1 scorers are post-action diagnostics: they can use action counts from the
current turn. V2 removes current action counts and next-step deltas, then adds a
single coarse candidate_action feature inferred from the observed replay action.

This is still observational data, not a true counterfactual model. It is useful
for comparing risk among action classes in similar states before any runtime
gate or reward integration.
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

from train_strategy_label_scorers import numeric_eval_side


STATE_FEATURES = [
    "map_size",
    "turn",
    "turns_remaining",
    "night_cycle",
    "cycle_turn",
    "pre_night",
    "is_night",
    "turns_to_night",
    "team",
    "eval_side_numeric",
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
    "city_tiles_delta_10",
    "city_tiles_growth_10",
    "workers_delta_10",
    "workers_growth_10",
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
]

ACTION_FEATURES = [
    "candidate_is_no_expand",
    "candidate_is_bw",
    "candidate_is_bcity",
    "candidate_is_research",
    "candidate_is_low_fuel_bw",
    "candidate_is_low_fuel_bcity",
]

FEATURES = STATE_FEATURES + ACTION_FEATURES

SCORER_SPECS = {
    "candidate_risk_city_loss_20": {
        "label": "risk_city_loss_20",
        "threshold": 0.35,
        "max_pos_weight": 8.0,
    },
    "candidate_risk_big_loss_20": {
        "label": "risk_big_loss_20",
        "threshold": 0.25,
        "max_pos_weight": 12.0,
    },
    "candidate_error_failed_big_loss": {
        "label": "error_failed_with_big_loss",
        "threshold": 0.20,
        "max_pos_weight": 16.0,
    },
    "candidate_safe_expansion_success_40": {
        "label": "expansion_success_40",
        "threshold": 0.45,
        "max_pos_weight": 8.0,
    },
    "candidate_success_stable_scale": {
        "label": "success_stable_scale",
        "threshold": 0.50,
        "max_pos_weight": 6.0,
    },
}


def stable_group_is_val(file_value: str, episode_id: str, team: object, val_fraction: float) -> bool:
    key = f"{file_value}:{episode_id}:{team}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def infer_candidate_action(data: pd.DataFrame) -> pd.Series:
    bcity = pd.to_numeric(data.get("bcity_actions", 0), errors="coerce").fillna(0)
    bw = pd.to_numeric(data.get("bw_actions", 0), errors="coerce").fillna(0)
    research = pd.to_numeric(data.get("research_actions", 0), errors="coerce").fillna(0)
    action = pd.Series("no_expand", index=data.index)
    action = action.mask(research > 0, "research")
    action = action.mask(bw > 0, "bw")
    action = action.mask(bcity > 0, "bcity")
    return action


def add_candidate_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["candidate_action"] = infer_candidate_action(frame)
    for action in ["no_expand", "bw", "bcity", "research"]:
        frame[f"candidate_is_{action}"] = (frame["candidate_action"] == action).astype(int)
    low_bw = pd.to_numeric(frame.get("bw_low_fuel_lt5_actions", 0), errors="coerce").fillna(0)
    low_bcity = pd.to_numeric(frame.get("bcity_adjacent_low_fuel_lt5_actions", 0), errors="coerce").fillna(0)
    frame["candidate_is_low_fuel_bw"] = ((frame["candidate_action"] == "bw") & (low_bw > 0)).astype(int)
    frame["candidate_is_low_fuel_bcity"] = ((frame["candidate_action"] == "bcity") & (low_bcity > 0)).astype(int)
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

    for column in FEATURES + ["sample_weight", "file", "episode_id", "team"]:
        if column not in data:
            data[column] = 0.0
    for spec in SCORER_SPECS.values():
        if spec["label"] not in data:
            data[spec["label"]] = 0
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data["sample_weight"] = pd.to_numeric(data["sample_weight"], errors="coerce").fillna(1.0).clip(lower=0.1, upper=10.0)
    for spec in SCORER_SPECS.values():
        data[spec["label"]] = pd.to_numeric(data[spec["label"]], errors="coerce").fillna(0).astype(int)
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
    for threshold in [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        item = classifier_metrics(y_true, prob, threshold)
        item["threshold"] = threshold
        rows.append(item)
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


def action_breakdown(data: pd.DataFrame, label: str, prob: np.ndarray, threshold: float) -> dict:
    frame = data[["candidate_action", label]].copy()
    frame["prob"] = prob
    frame["alert"] = frame["prob"] >= threshold
    rows = {}
    for action, group in frame.groupby("candidate_action"):
        rows[str(action)] = {
            "rows": int(len(group)),
            "positive_rate": float(group[label].mean()),
            "mean_prob": float(group["prob"].mean()),
            "alert_rate": float(group["alert"].mean()),
        }
    return rows


def train_one(name: str, spec: dict, data: pd.DataFrame, args: argparse.Namespace) -> dict:
    label = spec["label"]
    train_df = data[~data["is_val"]].copy()
    val_df = data[data["is_val"]].copy()
    if train_df.empty or val_df.empty:
        raise ValueError(f"{name}: train/val split is empty.")
    if train_df[label].nunique() < 2 or val_df[label].nunique() < 2:
        raise ValueError(f"{name}: label has fewer than two classes.")
    pos = max(int(train_df[label].sum()), 1)
    neg = max(len(train_df) - pos, 1)
    scale_pos_weight = min(neg / pos, float(spec["max_pos_weight"]))

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
        train_df[label],
        sample_weight=train_df["sample_weight"],
        eval_set=[(val_df[FEATURES], val_df[label])],
        eval_sample_weight=[val_df["sample_weight"]],
        eval_metric="auc",
        callbacks=callbacks,
    )

    train_prob = model.predict_proba(train_df[FEATURES])[:, 1]
    val_prob = model.predict_proba(val_df[FEATURES])[:, 1]
    model_dir = args.output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": spec["threshold"],
            "label_column": label,
            "candidate_actions": ["no_expand", "bw", "bcity", "research"],
            "map_size": args.map_size,
        },
        model_dir / f"{name}_lgbm.joblib",
    )
    write_importance(model, model_dir / "feature_importance.csv")
    summary = {
        "name": name,
        "label_column": label,
        "threshold": spec["threshold"],
        "rows": int(len(data)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "positive_rows": int(data[label].sum()),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "train": classifier_metrics(train_df[label].to_numpy(), train_prob, spec["threshold"]),
        "val": classifier_metrics(val_df[label].to_numpy(), val_prob, spec["threshold"]),
        "val_threshold_table": threshold_table(val_df[label].to_numpy(), val_prob),
        "val_by_candidate_action": action_breakdown(val_df, label, val_prob, spec["threshold"]),
        "model_path": str(model_dir / f"{name}_lgbm.joblib"),
        "feature_importance_path": str(model_dir / "feature_importance.csv"),
    }
    (model_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Strategy Candidate Scorer v2",
        "",
        f"- input: `{summary['input_csv']}`",
        f"- rows: {summary['rows']}",
        f"- map_size: {summary['map_size']}",
        "",
        "| scorer | label | rows | pos rate | AUC | AP | precision | recall | F1 | threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["scorers"].items():
        val = item["val"]
        lines.append(
            f"| {name} | {item['label_column']} | {item['rows']} | "
            f"{val['positive_rate']:.3f} | {val['auc']:.3f} | {val['average_precision']:.3f} | "
            f"{val['precision']:.3f} | {val['recall']:.3f} | {val['f1']:.3f} | {item['threshold']:.2f} |"
        )
    lines.append("")
    lines.append("## Candidate Action Distribution")
    lines.append("")
    lines.append("| action | rows |")
    lines.append("|---|---:|")
    for action, count in summary["candidate_action_counts"].items():
        lines.append(f"| {action} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate-action LightGBM strategy scorers v2.")
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/strategy_candidate_scorers_v2_16"))
    parser.add_argument("--map-size", type=int, default=16)
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
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()

    data = load_data(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorers = {}
    for name, spec in SCORER_SPECS.items():
        print(f"training {name} ({spec['label']})...")
        scorers[name] = train_one(name, spec, data, args)
        print(json.dumps(scorers[name]["val"], indent=2, ensure_ascii=False))
    summary = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "map_size": args.map_size,
        "rows": int(len(data)),
        "features": FEATURES,
        "candidate_action_counts": data["candidate_action"].value_counts().to_dict(),
        "scorers": scorers,
        "note": "V2 is observational action-conditioned scoring, not true counterfactual causal inference.",
    }
    (args.output_dir / "candidate_scorer_training_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "candidate_scorer_training_report.md", summary)
    print(f"report: {args.output_dir / 'candidate_scorer_training_report.json'}")
    print(f"markdown: {args.output_dir / 'candidate_scorer_training_report.md'}")


if __name__ == "__main__":
    main()
