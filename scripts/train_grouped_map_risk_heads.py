#!/usr/bin/env python3
"""Train per-map risk heads with replay-grouped splits and raw calibration.

The input is the unified strategy-label CSV produced from Lux replays. Rows
from one replay (or one deployed seed across opponents/sides) are never split
between train, validation, target evaluation, and calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_strategy_label_scorers import FEATURES as BASE_FEATURES  # noqa: E402


LABEL = "risk_big_loss_20"
MAP_SIZES = (12, 16, 24, 32)
LEAKY_FEATURES = {"city_tiles_delta_next", "units_delta_next", "research_delta_next"}
FEATURES = [feature for feature in BASE_FEATURES if feature not in LEAKY_FEATURES]
INPUT_FEATURES = [feature for feature in FEATURES if feature != "eval_side_numeric"]
METADATA_COLUMNS = [
    "file",
    "source_kind",
    "source_format",
    "episode_id",
    "map_size",
    "turn",
    "team",
    "eval_side",
    "sample_weight",
    LABEL,
]
USE_COLUMNS = list(dict.fromkeys(METADATA_COLUMNS + INPUT_FEATURES))
DEPLOYED_SEED_RE = re.compile(r"_(\d+)_p[01]\.json$", re.IGNORECASE)


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def numeric_eval_side(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.replace(".0", "", regex=False)
    return pd.to_numeric(text.replace({"": "-1"}), errors="coerce").fillna(-1).astype(float)


def source_class(frame: pd.DataFrame, deployed_marker: str) -> pd.Series:
    files = frame["file"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.lower()
    kinds = frame["source_kind"].fillna("").astype(str).str.lower()
    marker = deployed_marker.replace("\\", "/").lower()
    raw = kinds.eq("official_or_downloaded") | files.str.contains("dataset/raw/data", regex=False)
    deployed = files.str.contains(marker, regex=False)
    return pd.Series(np.where(raw, "raw", np.where(deployed, "deployed", "exclude")), index=frame.index)


def replay_group_key(row: pd.Series) -> str:
    source = str(row["source_class"])
    map_size = int(row["map_size"])
    file_text = str(row["file"]).replace("\\", "/")
    if source == "deployed":
        match = DEPLOYED_SEED_RE.search(file_text)
        if match:
            return f"deployed:map{map_size}:seed:{match.group(1)}"
    episode = str(row.get("episode_id", "")).strip()
    if source == "raw" and episode and episode.lower() not in {"nan", "none"}:
        return f"raw:map{map_size}:episode:{episode}"
    return f"{source}:map{map_size}:replay:{file_text.lower()}"


def read_inputs(paths: Iterable[Path], args: argparse.Namespace) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns
        missing = sorted(set(USE_COLUMNS) - set(header))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for chunk in pd.read_csv(path, usecols=USE_COLUMNS, chunksize=args.csv_chunk_size, low_memory=False):
            chunk["map_size"] = pd.to_numeric(chunk["map_size"], errors="coerce").fillna(0).astype(int)
            chunk["turn"] = pd.to_numeric(chunk["turn"], errors="coerce").fillna(-1).astype(int)
            chunk = chunk[chunk["map_size"].isin(MAP_SIZES)]
            chunk = chunk[(chunk["turn"] >= 0) & (chunk["turn"] % args.frame_stride == 0)]
            if chunk.empty:
                continue
            chunk["source_class"] = source_class(chunk, args.deployed_path_marker)
            chunk = chunk[chunk["source_class"].isin(["raw", "deployed"])]
            if not chunk.empty:
                chunks.append(chunk)
    if not chunks:
        raise ValueError("No raw or deployed rows were loaded from the input CSV files.")
    data = pd.concat(chunks, ignore_index=True)
    data["eval_side_numeric"] = numeric_eval_side(data["eval_side"])
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0).astype(np.float32)
    data[LABEL] = pd.to_numeric(data[LABEL], errors="coerce").fillna(0).astype(np.int8)
    data["sample_weight"] = pd.to_numeric(data["sample_weight"], errors="coerce").fillna(1.0).clip(0.1, 10.0)
    data["group_key"] = data.apply(replay_group_key, axis=1)
    data = data.drop_duplicates(subset=["file", "team", "turn"], keep="last").reset_index(drop=True)
    return data


def allocate_groups(groups: list[str], split_counts: list[tuple[str, int]]) -> dict[str, str]:
    ordered = sorted(groups, key=lambda value: (stable_hash(value), value))
    allocation: dict[str, str] = {}
    cursor = 0
    for split, count in split_counts:
        for group in ordered[cursor: cursor + count]:
            allocation[group] = split
        cursor += count
    if cursor != len(ordered):
        raise AssertionError(f"Group allocation mismatch: allocated={cursor}, groups={len(ordered)}")
    return allocation


def three_way_counts(n: int, calibration_fraction: float, validation_fraction: float) -> list[tuple[str, int]]:
    if n < 3:
        raise ValueError(f"Raw source needs at least 3 replay groups, found {n}.")
    calibration = max(1, int(round(n * calibration_fraction)))
    validation = max(1, int(round(n * validation_fraction)))
    if calibration + validation >= n:
        calibration = validation = 1
    train = n - validation - calibration
    return [("train", train), ("validation", validation), ("calibration", calibration)]


def two_way_counts(n: int, evaluation_fraction: float) -> list[tuple[str, int]]:
    if n < 2:
        return [("target_train", n), ("target_evaluation", 0)]
    evaluation = max(1, int(round(n * evaluation_fraction)))
    evaluation = min(evaluation, n - 1)
    return [("target_train", n - evaluation), ("target_evaluation", evaluation)]


def assign_splits(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    allocations: dict[str, str] = {}
    for map_size in MAP_SIZES:
        map_data = data[data["map_size"] == map_size]
        raw_groups = sorted(map_data.loc[map_data["source_class"] == "raw", "group_key"].unique())
        deployed_groups = sorted(map_data.loc[map_data["source_class"] == "deployed", "group_key"].unique())
        allocations.update(allocate_groups(
            raw_groups,
            three_way_counts(len(raw_groups), args.calibration_fraction, args.validation_fraction),
        ))
        allocations.update(allocate_groups(
            deployed_groups,
            two_way_counts(len(deployed_groups), args.target_evaluation_fraction),
        ))
    data = data.copy()
    data["split"] = data["group_key"].map(allocations)
    if data["split"].isna().any():
        raise AssertionError("Some rows did not receive a replay-group split.")
    source_multiplier = np.where(data["source_class"].eq("deployed"), args.deployed_weight, 1.0)
    source_multiplier = np.where(
        data["source_class"].eq("deployed") & data["map_size"].eq(32),
        args.deployed_32_weight,
        source_multiplier,
    )
    data["training_weight"] = data["sample_weight"].to_numpy(dtype=float) * source_multiplier
    group_manifest = (
        data.groupby(["map_size", "source_class", "group_key", "split"], as_index=False)
        .agg(
            rows=(LABEL, "size"),
            positive_rows=(LABEL, "sum"),
            replay_files=("file", "nunique"),
            weight_mean=("training_weight", "mean"),
        )
    )
    return data, group_manifest


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    labels = labels.astype(int)
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    return {
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if labels.sum() else None,
        "threshold": float(threshold),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def select_precision_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_precision: float,
    min_alerts: int,
) -> tuple[float, bool, dict]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_labels = labels[order].astype(int)
    cumulative_tp = np.cumsum(sorted_labels)
    alerts = np.arange(1, len(sorted_labels) + 1)
    precision = cumulative_tp / alerts
    total_positives = max(int(sorted_labels.sum()), 1)
    recall = cumulative_tp / total_positives
    tie_end = np.r_[sorted_probability[:-1] != sorted_probability[1:], True]
    eligible = tie_end & (alerts >= min_alerts) & (precision >= target_precision)
    if np.any(eligible):
        eligible_indices = np.flatnonzero(eligible)
        best_index = eligible_indices[np.argmax(recall[eligible_indices])]
        threshold = float(sorted_probability[best_index])
        return threshold, True, binary_metrics(labels, probabilities, threshold)
    threshold = 1.0
    return threshold, False, binary_metrics(labels, probabilities, threshold)


def write_pr_artifacts(
    map_dir: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    target_precision: float,
) -> None:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    rows = [
        {"threshold": float(value), "precision": float(precision[i]), "recall": float(recall[i])}
        for i, value in enumerate(thresholds)
    ]
    rows.append({"threshold": None, "precision": float(precision[-1]), "recall": float(recall[-1])})
    pd.DataFrame(rows).to_csv(map_dir / "precision_recall_curve.csv", index=False, encoding="utf-8")

    fig, axis = plt.subplots(figsize=(6.4, 4.8), dpi=140)
    axis.plot(recall, precision, color="#176B87", linewidth=2)
    selected = binary_metrics(labels, probabilities, threshold)
    axis.scatter([selected["recall"]], [selected["precision"]], color="#C73E1D", s=45, zorder=3)
    axis.axhline(target_precision, color="#555555", linestyle="--", linewidth=1)
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(map_dir / "precision_recall_curve.png")
    plt.close(fig)


def split_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0, "groups": 0, "positive_rows": 0, "positive_rate": 0.0}
    positives = int(frame[LABEL].sum())
    return {
        "rows": int(len(frame)),
        "groups": int(frame["group_key"].nunique()),
        "replays": int(frame["file"].nunique()),
        "positive_rows": positives,
        "positive_rate": float(positives / len(frame)),
        "weight_mean": float(frame["training_weight"].mean()),
    }


def train_map_head(map_size: int, data: pd.DataFrame, args: argparse.Namespace) -> dict:
    map_data = data[data["map_size"] == map_size].copy()
    raw_train = map_data[map_data["split"] == "train"]
    target_train = map_data[map_data["split"] == "target_train"]
    train = pd.concat([raw_train, target_train], ignore_index=True)
    validation = map_data[map_data["split"] == "validation"]
    calibration = map_data[map_data["split"] == "calibration"]
    target_evaluation = map_data[map_data["split"] == "target_evaluation"]
    for name, frame in (("train", train), ("validation", validation), ("calibration", calibration)):
        if frame.empty or frame[LABEL].nunique() < 2:
            raise ValueError(f"map {map_size}: {name} split is empty or has fewer than two label classes")

    positives = max(int(train[LABEL].sum()), 1)
    negatives = max(len(train) - positives, 1)
    scale_pos_weight = min(negatives / positives, args.max_pos_weight)
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
        random_state=args.seed + map_size,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    model.fit(
        train[FEATURES],
        train[LABEL],
        sample_weight=train["training_weight"],
        eval_set=[(validation[FEATURES], validation[LABEL])],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(args.log_period),
        ],
    )
    calibration_prob = model.predict_proba(calibration[FEATURES])[:, 1]
    threshold, achieved, selected_metrics = select_precision_threshold(
        calibration[LABEL].to_numpy(),
        calibration_prob,
        args.target_precision,
        args.min_calibration_alerts,
    )

    map_dir = args.output_dir / f"map_{map_size}"
    map_dir.mkdir(parents=True, exist_ok=True)
    package = {
        "model": model,
        "features": FEATURES,
        "label": LABEL,
        "map_size": map_size,
        "risk_threshold": threshold,
        "target_precision": args.target_precision,
        "grouped_split": True,
    }
    joblib.dump(package, map_dir / "risk_big_loss_20_lgbm.joblib")
    write_pr_artifacts(
        map_dir,
        calibration[LABEL].to_numpy(),
        calibration_prob,
        threshold,
        args.target_precision,
    )

    evaluations = {}
    for split_name, frame in (
        ("validation", validation),
        ("calibration", calibration),
        ("target_evaluation", target_evaluation),
    ):
        if frame.empty:
            evaluations[split_name] = None
            continue
        probability = model.predict_proba(frame[FEATURES])[:, 1]
        evaluations[split_name] = binary_metrics(frame[LABEL].to_numpy(), probability, threshold)
        scored = frame[["file", "group_key", "source_class", "team", "turn", LABEL]].copy()
        scored["risk_score"] = probability
        scored["risk_threshold"] = threshold
        scored["risk_alert"] = (probability >= threshold).astype(int)
        scored.to_csv(map_dir / f"{split_name}_scores.csv", index=False, encoding="utf-8")

    split_counts = {
        split: split_summary(map_data[map_data["split"] == split])
        for split in ("train", "validation", "calibration", "target_train", "target_evaluation")
    }
    threshold_artifact = {
        "map_size": map_size,
        "label": LABEL,
        "risk_threshold": float(threshold),
        "target_precision": float(args.target_precision),
        "target_precision_achieved": bool(achieved),
        "calibration_metrics": selected_metrics,
        "calibration_source": "raw_only",
        "minimum_alerts": int(args.min_calibration_alerts),
    }
    (map_dir / "risk_threshold.yaml").write_text(
        yaml.safe_dump(threshold_artifact, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    report = {
        **threshold_artifact,
        "model_path": str(map_dir / "risk_big_loss_20_lgbm.joblib"),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "scale_pos_weight": float(scale_pos_weight),
        "features": FEATURES,
        "excluded_future_features": sorted(LEAKY_FEATURES),
        "splits": split_counts,
        "evaluation": evaluations,
    }
    (map_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def assert_no_group_leakage(data: pd.DataFrame) -> None:
    split_counts = data.groupby("group_key")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        raise AssertionError(f"Replay groups assigned to multiple splits: {leaking.index[:10].tolist()}")
    calibration_sources = set(data.loc[data["split"] == "calibration", "source_class"])
    if calibration_sources != {"raw"}:
        raise AssertionError(f"Calibration must be raw-only, found sources={sorted(calibration_sources)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train replay-grouped, per-map Lux risk heads.")
    parser.add_argument("--input-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spatial_risk_map_calibration_v1"))
    parser.add_argument("--deployed-path-marker", default="spatial_risk_deployed_replays")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--csv-chunk-size", type=int, default=100000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--target-evaluation-fraction", type=float, default=0.25)
    parser.add_argument("--deployed-weight", type=float, default=1.5)
    parser.add_argument("--deployed-32-weight", type=float, default=0.25)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--min-calibration-alerts", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=12.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_inputs(args.input_csv, args)
    data, group_manifest = assign_splits(data, args)
    assert_no_group_leakage(data)
    group_manifest.to_csv(args.output_dir / "group_split_manifest.csv", index=False, encoding="utf-8")

    reports = {}
    for map_size in MAP_SIZES:
        print(f"training map {map_size} risk head...", flush=True)
        reports[str(map_size)] = train_map_head(map_size, data, args)
        calibration = reports[str(map_size)]["evaluation"]["calibration"]
        print(
            f"map={map_size} threshold={reports[str(map_size)]['risk_threshold']:.6f} "
            f"precision={calibration['precision']:.4f} recall={calibration['recall']:.4f}",
            flush=True,
        )

    summary = {
        "input_csvs": [str(path) for path in args.input_csv],
        "output_dir": str(args.output_dir),
        "label": LABEL,
        "features": FEATURES,
        "frame_stride": args.frame_stride,
        "grouped_by_replay_or_deployed_seed": True,
        "calibration_source": "raw_only",
        "deployed_weight": args.deployed_weight,
        "deployed_32_weight": args.deployed_32_weight,
        "target_precision": args.target_precision,
        "maps": reports,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    threshold_bundle = {
        "label": LABEL,
        "target_precision": float(args.target_precision),
        "maps": {
            int(map_size): {
                "risk_threshold": float(report["risk_threshold"]),
                "target_precision_achieved": bool(report["target_precision_achieved"]),
                "calibration_rows": int(report["evaluation"]["calibration"]["rows"]),
                "calibration_positive_rows": int(report["evaluation"]["calibration"]["positive_rows"]),
            }
            for map_size, report in reports.items()
        },
    }
    (args.output_dir / "risk_thresholds.yaml").write_text(
        yaml.safe_dump(threshold_bundle, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    sample_rows = []
    for map_size, report in reports.items():
        for split, counts in report["splits"].items():
            sample_rows.append({"map_size": int(map_size), "split": split, **counts})
    pd.DataFrame(sample_rows).to_csv(args.output_dir / "sample_counts.csv", index=False, encoding="utf-8")
    print(f"summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
