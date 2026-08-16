#!/usr/bin/env python3
"""Train LightGBM scorers from the unified Lux strategy-label dataset.

These scorers are diagnostic models. They predict risk/error/safe-expansion/
success labels from current and past-observed features only, avoiding future and
final outcome leakage.
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
    "action_count",
    "move_actions",
    "transfer_actions",
    "pillage_actions",
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
    "city_tiles_delta_next",
    "units_delta_next",
    "research_delta_next",
]


SCORER_SPECS = {
    "risk_city_loss_20": {
        "label": "risk_city_loss_20",
        "threshold": 0.35,
        "max_pos_weight": 8.0,
        "row_filter": "all",
    },
    "risk_big_loss_20": {
        "label": "risk_big_loss_20",
        "threshold": 0.25,
        "max_pos_weight": 12.0,
        "row_filter": "all",
    },
    "error_failed_big_loss": {
        "label": "error_failed_with_big_loss",
        "threshold": 0.20,
        "max_pos_weight": 16.0,
        "row_filter": "all",
    },
    "safe_expansion_success_40": {
        "label": "expansion_success_40",
        "threshold": 0.45,
        "max_pos_weight": 8.0,
        "row_filter": "expansion_candidates",
    },
    "success_stable_scale": {
        "label": "success_stable_scale",
        "threshold": 0.50,
        "max_pos_weight": 6.0,
        "row_filter": "all",
    },
}


def stable_group_is_val(file_value: str, episode_id: str, team: object, val_fraction: float) -> bool:
    key = f"{file_value}:{episode_id}:{team}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def numeric_eval_side(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.replace(".0", "", regex=False)
    return pd.to_numeric(text.replace({"": "-1"}), errors="coerce").fillna(-1).astype(float)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    usecols = None
    data = pd.read_csv(args.input_csv, usecols=usecols, low_memory=False)
    if args.map_size:
        data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == args.map_size].copy()
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if data.empty:
        raise ValueError("No strategy label rows loaded.")

    data["eval_side_numeric"] = numeric_eval_side(data.get("eval_side", pd.Series([""] * len(data))))
    for column in FEATURES + ["sample_weight", "file", "episode_id", "team"]:
        if column not in data.columns:
            data[column] = 0.0
    for spec in SCORER_SPECS.values():
        if spec["label"] not in data.columns:
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


def apply_row_filter(data: pd.DataFrame, row_filter: str) -> pd.DataFrame:
    if row_filter == "all":
        return data
    if row_filter == "expansion_candidates":
        mask = (
            (pd.to_numeric(data.get("expansion_taken", 0), errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(data.get("expansion_safe_window_proxy", 0), errors="coerce").fillna(0) > 0)
            | (
                (pd.to_numeric(data.get("unit_cap_margin", 0), errors="coerce").fillna(0) > 0)
                & (pd.to_numeric(data.get("turns_remaining", 0), errors="coerce").fillna(0) >= 40)
            )
        )
        return data[mask].copy()
    raise ValueError(f"Unknown row_filter: {row_filter}")


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


def train_one(name: str, spec: dict, data: pd.DataFrame, args: argparse.Namespace) -> dict:
    label = spec["label"]
    model_data = apply_row_filter(data, spec["row_filter"])
    train_df = model_data[~model_data["is_val"]].copy()
    val_df = model_data[model_data["is_val"]].copy()
    if train_df.empty or val_df.empty:
        raise ValueError(f"{name}: train/validation split is empty.")
    if train_df[label].nunique() < 2 or val_df[label].nunique() < 2:
        raise ValueError(f"{name}: label has fewer than two classes in train or validation.")

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
            "row_filter": spec["row_filter"],
            "map_size": args.map_size,
        },
        model_dir / f"{name}_lgbm.joblib",
    )
    write_importance(model, model_dir / "feature_importance.csv")
    summary = {
        "name": name,
        "label_column": label,
        "row_filter": spec["row_filter"],
        "threshold": spec["threshold"],
        "rows": int(len(model_data)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "positive_rows": int(model_data[label].sum()),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "train": classifier_metrics(train_df[label].to_numpy(), train_prob, spec["threshold"]),
        "val": classifier_metrics(val_df[label].to_numpy(), val_prob, spec["threshold"]),
        "val_threshold_table": threshold_table(val_df[label].to_numpy(), val_prob),
        "model_path": str(model_dir / f"{name}_lgbm.joblib"),
        "feature_importance_path": str(model_dir / "feature_importance.csv"),
    }
    (model_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Strategy Label Scorer Training",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified LightGBM strategy-label scorers.")
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/strategy_label_scorers_v1_16"))
    parser.add_argument("--map-size", type=int, default=16, help="Filter to one map size. Use 0 to train on all map sizes.")
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
    parser.add_argument("--seed", type=int, default=41)
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
        "scorers": scorers,
    }
    (args.output_dir / "strategy_scorer_training_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "strategy_scorer_training_report.md", summary)
    print(f"report: {args.output_dir / 'strategy_scorer_training_report.json'}")
    print(f"markdown: {args.output_dir / 'strategy_scorer_training_report.md'}")


if __name__ == "__main__":
    main()
