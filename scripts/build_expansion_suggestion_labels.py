#!/usr/bin/env python3
"""Build conservative expansion suggestion labels from candidate-action scores.

The labels are intentionally soft. They identify healthy missed bcity windows
that can be used later for reweighting or auxiliary training, not direct action
overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "state_id",
    "rank",
    "turn",
    "cycle_turn",
    "turns_to_night",
    "city_tiles",
    "workers",
    "research",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "low_fuel_city_lt5",
    "future_team_loss_20",
    "final_city_tile_margin",
    "bad_score_delta",
    "bcity_big_risk",
    "bcity_error_risk",
    "bcity_safe_expansion",
    "bcity_success",
    "no_expand_big_risk",
    "no_expand_success",
]

FIELDNAMES = [
    "label_type",
    "label_version",
    "source_file",
    "source_opponent",
    "eval_side",
    "team",
    "team_name",
    "opponent_name",
    "rank",
    "turn",
    "phase",
    "cycle_turn",
    "turns_to_night",
    "city_tiles",
    "workers",
    "research",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "low_fuel_city_lt5",
    "future_team_loss_20",
    "final_city_tile_margin",
    "actual_action",
    "suggested_action_type",
    "expansion_positive_label",
    "expansion_positive_weight",
    "priority_bucket",
    "bcity_safe_expansion",
    "bcity_big_risk",
    "bcity_error_risk",
    "bcity_success",
    "no_expand_big_risk",
    "no_expand_success",
    "bad_score_delta",
    "reason",
]


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    for column in ["suggestion", "file", "source_opponent", "phase", "actual_action"]:
        if column in data:
            data[column] = data[column].fillna("").astype(str)
    return data


def priority_bucket(row: pd.Series, args: argparse.Namespace) -> str:
    if int(row["rank"]) == 2 and row["turn"] >= args.late_turn and row["final_city_tile_margin"] < 0:
        return "loss_late_scale_gap"
    if int(row["rank"]) == 2 and row["final_city_tile_margin"] < 0:
        return "loss_scale_gap"
    if row["turn"] >= args.late_turn:
        return "late_safe_growth"
    return "safe_growth"


def label_weight(row: pd.Series, args: argparse.Namespace) -> float:
    weight = args.base_weight
    if int(row["rank"]) == 2:
        weight += args.loss_bonus
    if row["turn"] >= args.late_turn:
        weight += args.late_bonus
    if row["final_city_tile_margin"] < 0:
        weight += min(abs(float(row["final_city_tile_margin"])) / args.margin_scale, args.max_margin_bonus)
    if row["p25_city_fuel_turns"] >= args.strong_p25_fuel:
        weight += args.strong_buffer_bonus
    if row["bcity_safe_expansion"] >= args.strong_safe_threshold:
        weight += args.strong_safe_bonus
    return round(min(weight, args.max_weight), 4)


def build_labels(data: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    missed = data[data["suggestion"] == "missed_safe_bcity_window"].copy()
    healthy = missed[
        (missed["p25_city_fuel_turns"] >= args.min_p25_fuel)
        & (missed["min_city_fuel_turns"] >= args.min_city_fuel)
        & (missed["bcity_big_risk"] <= args.max_bcity_big_risk)
        & (missed["bcity_error_risk"] <= args.max_bcity_error_risk)
        & (missed["future_team_loss_20"] <= args.max_future_loss20)
        & (missed["bcity_safe_expansion"] >= args.min_bcity_safe)
        & (missed["turn"] >= args.min_turn)
        & (missed["turn"] <= args.max_turn)
    ].copy()

    rows = []
    for _, row in healthy.iterrows():
        bucket = priority_bucket(row, args)
        rows.append(
            {
                "label_type": "expansion_suggestion",
                "label_version": args.label_version,
                "source_file": row.get("file", ""),
                "source_opponent": row.get("source_opponent", ""),
                "eval_side": row.get("eval_side", ""),
                "team": row.get("team", ""),
                "team_name": row.get("team_name", ""),
                "opponent_name": row.get("opponent_name", ""),
                "rank": int(row.get("rank", 0)),
                "turn": int(row.get("turn", 0)),
                "phase": row.get("phase", ""),
                "cycle_turn": int(row.get("cycle_turn", 0)),
                "turns_to_night": int(row.get("turns_to_night", 0)),
                "city_tiles": int(row.get("city_tiles", 0)),
                "workers": int(row.get("workers", 0)),
                "research": int(row.get("research", 0)),
                "min_city_fuel_turns": float(row.get("min_city_fuel_turns", 0.0)),
                "p25_city_fuel_turns": float(row.get("p25_city_fuel_turns", 0.0)),
                "low_fuel_city_lt5": int(row.get("low_fuel_city_lt5", 0)),
                "future_team_loss_20": float(row.get("future_team_loss_20", 0.0)),
                "final_city_tile_margin": float(row.get("final_city_tile_margin", 0.0)),
                "actual_action": row.get("actual_action", ""),
                "suggested_action_type": "bcity",
                "expansion_positive_label": 1,
                "expansion_positive_weight": label_weight(row, args),
                "priority_bucket": bucket,
                "bcity_safe_expansion": float(row.get("bcity_safe_expansion", 0.0)),
                "bcity_big_risk": float(row.get("bcity_big_risk", 0.0)),
                "bcity_error_risk": float(row.get("bcity_error_risk", 0.0)),
                "bcity_success": float(row.get("bcity_success", 0.0)),
                "no_expand_big_risk": float(row.get("no_expand_big_risk", 0.0)),
                "no_expand_success": float(row.get("no_expand_success", 0.0)),
                "bad_score_delta": float(row.get("bad_score_delta", 0.0)),
                "reason": "healthy missed safe bcity window",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIELDNAMES).to_csv(path, index=False, encoding="utf-8")


def group_summary(data: pd.DataFrame, key: str) -> dict:
    if data.empty or key not in data:
        return {}
    return (
        data.groupby(key)
        .agg(
            rows=("label_type", "size"),
            mean_weight=("expansion_positive_weight", "mean"),
            mean_turn=("turn", "mean"),
            mean_margin=("final_city_tile_margin", "mean"),
            mean_p25_fuel=("p25_city_fuel_turns", "mean"),
            mean_bcity_safe=("bcity_safe_expansion", "mean"),
            mean_bcity_big_risk=("bcity_big_risk", "mean"),
        )
        .round(6)
        .to_dict("index")
    )


def write_summary(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    data = pd.DataFrame(rows)
    summary = {
        "label_type": "expansion_suggestion",
        "label_version": args.label_version,
        "input": str(args.input),
        "rows": int(len(data)),
        "filters": {
            "min_p25_fuel": args.min_p25_fuel,
            "min_city_fuel": args.min_city_fuel,
            "max_bcity_big_risk": args.max_bcity_big_risk,
            "max_bcity_error_risk": args.max_bcity_error_risk,
            "max_future_loss20": args.max_future_loss20,
            "min_bcity_safe": args.min_bcity_safe,
            "min_turn": args.min_turn,
            "max_turn": args.max_turn,
        },
    }
    if not data.empty:
        summary.update(
            {
                "mean_weight": float(data["expansion_positive_weight"].mean()),
                "max_weight": float(data["expansion_positive_weight"].max()),
                "by_priority_bucket": group_summary(data, "priority_bucket"),
                "by_source_opponent": group_summary(data, "source_opponent"),
                "by_rank": group_summary(data, "rank"),
                "by_phase": group_summary(data, "phase"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative expansion suggestion labels.")
    parser.add_argument("--input", type=Path, default=Path("outputs/diagnostic_layer/best_agent_candidate_action_suggestions_v2_16/candidate_action_suggestions.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("dataset/processed/expansion_suggestion_labels_v1.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("dataset/processed/expansion_suggestion_labels_v1_summary.json"))
    parser.add_argument("--label-version", default="expansion_suggestion_v1")
    parser.add_argument("--min-p25-fuel", type=float, default=10.0)
    parser.add_argument("--min-city-fuel", type=float, default=3.0)
    parser.add_argument("--max-bcity-big-risk", type=float, default=0.12)
    parser.add_argument("--max-bcity-error-risk", type=float, default=0.05)
    parser.add_argument("--max-future-loss20", type=float, default=1.0)
    parser.add_argument("--min-bcity-safe", type=float, default=0.80)
    parser.add_argument("--min-turn", type=int, default=80)
    parser.add_argument("--max-turn", type=int, default=320)
    parser.add_argument("--late-turn", type=int, default=160)
    parser.add_argument("--base-weight", type=float, default=0.15)
    parser.add_argument("--loss-bonus", type=float, default=0.08)
    parser.add_argument("--late-bonus", type=float, default=0.04)
    parser.add_argument("--strong-buffer-bonus", type=float, default=0.03)
    parser.add_argument("--strong-p25-fuel", type=float, default=20.0)
    parser.add_argument("--strong-safe-bonus", type=float, default=0.03)
    parser.add_argument("--strong-safe-threshold", type=float, default=0.85)
    parser.add_argument("--margin-scale", type=float, default=80.0)
    parser.add_argument("--max-margin-bonus", type=float, default=0.08)
    parser.add_argument("--max-weight", type=float, default=0.35)
    args = parser.parse_args()

    data = load_data(args.input)
    rows = build_labels(data, args)
    write_csv(args.output_csv, rows)
    write_summary(args.summary_json, rows, args)
    print(f"rows: {len(rows)}")
    print(f"labels: {args.output_csv}")
    print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
