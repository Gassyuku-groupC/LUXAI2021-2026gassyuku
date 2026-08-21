#!/usr/bin/env python3
"""Validate unified strategy labels produced from Lux replay data.

The goal is not model training. This script checks whether labels are useful
diagnostic signals: risk/error labels should concentrate future city loss, while
expansion/success labels should correlate with safer growth and wins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "turn",
    "team",
    "map_size",
    "win_label",
    "loss_label",
    "city_tiles",
    "workers",
    "research",
    "fuel_turns_total",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "future_team_loss_10",
    "future_team_loss_20",
    "future_team_loss_30",
    "future_team_loss_40",
    "future_city_tiles_gain_20",
    "future_city_tiles_gain_40",
    "final_city_tiles",
    "final_opponent_city_tiles",
    "final_city_tile_margin",
    "risk_city_loss_20",
    "risk_big_loss_20",
    "error_failed_with_big_loss",
    "error_low_fuel_bw_then_loss",
    "error_low_fuel_bcity_then_loss",
    "error_late_research_then_loss",
    "error_scale_growth_fuel_drop_then_loss",
    "expansion_taken",
    "expansion_safe_20",
    "expansion_success_40",
    "expansion_failed_20",
    "expansion_safe_window_proxy",
    "success_win",
    "success_scale_advantage",
    "success_stable_scale",
    "sample_weight",
]

LABEL_COLUMNS = [
    "risk_city_loss_20",
    "risk_big_loss_20",
    "risk_low_fuel_pre_night",
    "risk_low_fuel_night",
    "risk_scale_without_buffer",
    "error_failed_with_big_loss",
    "error_low_fuel_bw_then_loss",
    "error_low_fuel_bcity_then_loss",
    "error_late_research_then_loss",
    "error_scale_growth_fuel_drop_then_loss",
    "expansion_taken",
    "expansion_safe_20",
    "expansion_success_40",
    "expansion_failed_20",
    "expansion_safe_window_proxy",
    "success_survived",
    "success_win",
    "success_scale_advantage",
    "success_stable_scale",
]


def load_dataset(path: Path, max_rows: int = 0) -> pd.DataFrame:
    data = pd.read_csv(path, nrows=max_rows or None, low_memory=False)
    for column in NUMERIC_COLUMNS + LABEL_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    for column in ["source_kind", "source_opponent", "eval_side", "team_name", "strategy_label", "phase"]:
        if column in data:
            data[column] = data[column].fillna("").astype(str)
    return data


def safe_float(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def summarize_group(data: pd.DataFrame) -> dict:
    if data.empty:
        return {
            "rows": 0,
            "win_rate": 0.0,
            "loss20_rate": 0.0,
            "big_loss20_rate": 0.0,
            "mean_loss20": 0.0,
            "mean_gain40": 0.0,
            "expansion_success40_rate": 0.0,
            "final_city_tile_margin": 0.0,
        }
    return {
        "rows": int(len(data)),
        "win_rate": safe_float(data["success_win"].mean()),
        "loss20_rate": safe_float((data["future_team_loss_20"] > 0).mean()),
        "big_loss20_rate": safe_float((data["future_team_loss_20"] >= 10).mean()),
        "mean_loss20": safe_float(data["future_team_loss_20"].mean()),
        "mean_gain40": safe_float(data["future_city_tiles_gain_40"].mean()),
        "expansion_success40_rate": safe_float(data["expansion_success_40"].mean()),
        "final_city_tile_margin": safe_float(data["final_city_tile_margin"].mean()),
    }


def grouped_summary(data: pd.DataFrame, key: str, min_rows: int) -> dict:
    if key not in data:
        return {}
    rows = {}
    for value, group in data.groupby(key, dropna=False):
        if len(group) >= min_rows:
            rows[str(value)] = summarize_group(group)
    return rows


def binary_label_lift(data: pd.DataFrame, label: str) -> dict:
    if label not in data:
        return {}
    positive = data[data[label] > 0]
    negative = data[data[label] <= 0]
    pos = summarize_group(positive)
    neg = summarize_group(negative)
    return {
        "label": label,
        "positive": pos,
        "negative": neg,
        "loss20_rate_lift": safe_float(pos["loss20_rate"] - neg["loss20_rate"]),
        "big_loss20_rate_lift": safe_float(pos["big_loss20_rate"] - neg["big_loss20_rate"]),
        "win_rate_lift": safe_float(pos["win_rate"] - neg["win_rate"]),
        "gain40_lift": safe_float(pos["mean_gain40"] - neg["mean_gain40"]),
    }


def contradiction_checks(data: pd.DataFrame) -> dict:
    checks = {}
    if "strategy_label" in data:
        error = data[data["strategy_label"] == "error"]
        expansion = data[data["strategy_label"] == "expansion_success"]
        success = data[data["strategy_label"] == "success"]
        risk = data[data["strategy_label"] == "risk"]
        checks["error_without_future_loss20_rate"] = safe_float((error["future_team_loss_20"] <= 0).mean()) if len(error) else 0.0
        checks["expansion_success_with_loss20_rate"] = safe_float((expansion["future_team_loss_20"] > 0).mean()) if len(expansion) else 0.0
        checks["success_rows_with_loss_label_rate"] = safe_float((success["loss_label"] > 0).mean()) if len(success) else 0.0
        checks["risk_rows_with_no_loss20_rate"] = safe_float((risk["future_team_loss_20"] <= 0).mean()) if len(risk) else 0.0
    if "expansion_safe_20" in data:
        safe_expansion = data[data["expansion_safe_20"] > 0]
        checks["expansion_safe20_with_loss20_rate"] = (
            safe_float((safe_expansion["future_team_loss_20"] > 0).mean()) if len(safe_expansion) else 0.0
        )
    return checks


def validation_verdict(summary: dict) -> dict:
    notes = []
    pass_checks = True

    by_strategy = summary.get("by_strategy_label", {})
    error = by_strategy.get("error", {})
    neutral = by_strategy.get("neutral", {})
    expansion = by_strategy.get("expansion_success", {})
    if error and neutral and error["loss20_rate"] <= neutral["loss20_rate"]:
        pass_checks = False
        notes.append("error label does not have higher future loss20 rate than neutral.")
    if expansion and expansion["loss20_rate"] > 0.20:
        notes.append("expansion_success contains a non-trivial loss20 tail; inspect thresholds before training.")
    if summary.get("contradictions", {}).get("expansion_safe20_with_loss20_rate", 0) > 0.02:
        pass_checks = False
        notes.append("expansion_safe_20 contradicts future loss labels too often.")

    if not notes:
        notes.append("label layer is directionally consistent for diagnostic use.")
    return {"passed_basic_consistency": pass_checks, "notes": notes}


def write_markdown(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# Strategy Label Validation")
    lines.append("")
    lines.append(f"- rows: {summary['rows']}")
    lines.append(f"- source: `{summary['input_csv']}`")
    lines.append(f"- verdict: `{summary['verdict']['passed_basic_consistency']}`")
    for note in summary["verdict"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## By Strategy Label")
    lines.append("")
    lines.append("| label | rows | win | loss20 | big_loss20 | mean_loss20 | gain40 | margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, item in summary["by_strategy_label"].items():
        lines.append(
            f"| {label} | {item['rows']} | {item['win_rate']:.3f} | "
            f"{item['loss20_rate']:.3f} | {item['big_loss20_rate']:.3f} | "
            f"{item['mean_loss20']:.3f} | {item['mean_gain40']:.3f} | "
            f"{item['final_city_tile_margin']:.3f} |"
        )
    lines.append("")
    lines.append("## Label Lift")
    lines.append("")
    lines.append("| label | pos rows | loss20 lift | big_loss20 lift | win lift | gain40 lift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in summary["label_lift"]:
        lines.append(
            f"| {item['label']} | {item['positive']['rows']} | "
            f"{item['loss20_rate_lift']:.3f} | {item['big_loss20_rate_lift']:.3f} | "
            f"{item['win_rate_lift']:.3f} | {item['gain40_lift']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strategy labels.")
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("dataset/processed/strategy_label_validation_v1.json"))
    parser.add_argument("--output-md", type=Path, default=Path("dataset/processed/strategy_label_validation_v1.md"))
    parser.add_argument("--min-group-rows", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    data = load_dataset(args.input_csv, args.max_rows)
    summary = {
        "input_csv": str(args.input_csv),
        "rows": int(len(data)),
        "by_strategy_label": grouped_summary(data, "strategy_label", 1),
        "by_source_kind": grouped_summary(data, "source_kind", args.min_group_rows),
        "by_eval_side": grouped_summary(data, "eval_side", args.min_group_rows),
        "by_source_opponent": grouped_summary(data, "source_opponent", args.min_group_rows),
        "by_phase": grouped_summary(data, "phase", args.min_group_rows),
        "label_lift": [binary_label_lift(data, label) for label in LABEL_COLUMNS if label in data],
        "contradictions": contradiction_checks(data),
    }
    summary["verdict"] = validation_verdict(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(f"rows: {len(data)}")
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_md}")
    print(json.dumps(summary["verdict"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
