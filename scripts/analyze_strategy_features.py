#!/usr/bin/env python3
"""Analyze strategy feature curves and loss correlations."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_METRICS = [
    "city_tiles",
    "cities",
    "workers",
    "worker_citytile_ratio",
    "research",
    "fuel_turns_total",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "low_fuel_city_lt5",
    "low_fuel_city_lt10",
    "bw_actions",
    "bw_low_fuel_lt3_actions",
    "bw_low_fuel_lt5_actions",
    "bcity_actions",
    "bcity_isolated_actions",
    "bcity_resource_near_actions",
    "bcity_adjacent_low_fuel_lt5_actions",
    "future_team_loss_10",
    "final_city_tiles",
]


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def as_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def load_rows(path: Path, map_size: int = 0, team_names: set[str] | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as in_file:
        for row in csv.DictReader(in_file):
            if map_size and as_int(row, "map_size") != map_size:
                continue
            if team_names and row.get("team_name", "") not in team_names:
                continue
            rows.append(row)
    return rows


def turn_bucket(turn: int) -> str:
    if turn < 40:
        return "000-039"
    if turn < 80:
        return "040-079"
    if turn < 120:
        return "080-119"
    if turn < 160:
        return "120-159"
    if turn < 240:
        return "160-239"
    if turn < 320:
        return "240-319"
    return "320-360"


def phase_bucket(row: dict) -> str:
    return str(row.get("phase") or "unknown")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict], label_fn: Callable[[dict], str], metrics: list[str]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[label_fn(row)].append(row)
    summaries = {}
    for label, group in grouped.items():
        item = {"n": len(group)}
        for metric in metrics:
            item[metric] = mean(as_float(row, metric) for row in group)
        item["loss_rate_10"] = mean(1.0 if as_float(row, "future_team_loss_10") > 0 else 0.0 for row in group)
        item["big_loss_rate_10"] = mean(1.0 if as_float(row, "future_team_loss_10") >= 5 else 0.0 for row in group)
        summaries[label] = item
    return summaries


def write_summary_table(out, title: str, summaries: dict[str, dict], metrics: list[str]) -> None:
    out.write(f"\n[{title}]\n")
    header = ["label", "n", *metrics, "loss_rate_10", "big_loss_rate_10"]
    out.write(",".join(header) + "\n")
    for label in sorted(summaries):
        item = summaries[label]
        values = [label, str(item["n"])]
        for metric in metrics:
            values.append(f"{item.get(metric, 0.0):.4f}")
        values.append(f"{item.get('loss_rate_10', 0.0):.4f}")
        values.append(f"{item.get('big_loss_rate_10', 0.0):.4f}")
        out.write(",".join(values) + "\n")


def diff_table(reference: dict[str, dict], candidate: dict[str, dict], metrics: list[str], min_n: int) -> list[dict]:
    rows = []
    for label, ref in reference.items():
        cand = candidate.get(label)
        if not cand or ref["n"] < min_n or cand["n"] < min_n:
            continue
        for metric in metrics:
            ref_value = ref.get(metric, 0.0)
            cand_value = cand.get(metric, 0.0)
            rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "reference": ref_value,
                    "candidate": cand_value,
                    "diff": cand_value - ref_value,
                    "abs_diff": abs(cand_value - ref_value),
                    "reference_n": ref["n"],
                    "candidate_n": cand["n"],
                }
            )
    rows.sort(key=lambda row: row["abs_diff"], reverse=True)
    return rows


def fuel_bucket(row: dict, key: str) -> str:
    value = as_float(row, key)
    if value < 3:
        return f"{key}<3"
    if value < 5:
        return f"{key}=3-5"
    if value < 10:
        return f"{key}=5-10"
    if value < 20:
        return f"{key}=10-20"
    return f"{key}>=20"


def ratio_bucket(row: dict) -> str:
    value = as_float(row, "worker_citytile_ratio")
    if value < 0.5:
        return "worker_citytile_ratio<0.5"
    if value < 0.8:
        return "worker_citytile_ratio=0.5-0.8"
    if value < 1.1:
        return "worker_citytile_ratio=0.8-1.1"
    return "worker_citytile_ratio>=1.1"


def action_bucket(row: dict, key: str) -> str:
    value = as_int(row, key)
    if value == 0:
        return f"{key}=0"
    if value == 1:
        return f"{key}=1"
    return f"{key}>=2"


def loss_correlation_tables(rows: list[dict], min_rows: int, metrics: list[str]) -> dict[str, dict[str, dict]]:
    specs = {
        "phase": phase_bucket,
        "turn_bucket": lambda row: turn_bucket(as_int(row, "turn")),
        "min_city_fuel_turns": lambda row: fuel_bucket(row, "min_city_fuel_turns"),
        "fuel_turns_total": lambda row: fuel_bucket(row, "fuel_turns_total"),
        "worker_citytile_ratio": ratio_bucket,
        "bw_low_fuel_lt3_actions": lambda row: action_bucket(row, "bw_low_fuel_lt3_actions"),
        "bw_low_fuel_lt5_actions": lambda row: action_bucket(row, "bw_low_fuel_lt5_actions"),
        "bcity_adjacent_low_fuel_lt5_actions": lambda row: action_bucket(row, "bcity_adjacent_low_fuel_lt5_actions"),
    }
    tables = {}
    for name, label_fn in specs.items():
        summary = summarize(rows, label_fn, metrics)
        tables[name] = {label: item for label, item in summary.items() if item["n"] >= min_rows}
    return tables


def parse_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-turn strategy features.")
    parser.add_argument("--reference", type=Path, default=Path("dataset/processed/strategy_features_top12.csv"))
    parser.add_argument("--candidate", type=Path, default=Path("outputs/risk_feature_logs/strategy_features_baseline_best_seed12345_16x16.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/risk_feature_logs/strategy_feature_analysis_16x16.txt"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--reference-teams", default="", help="Comma-separated reference team names.")
    parser.add_argument("--candidate-teams", default="", help="Comma-separated candidate team names.")
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--top-diffs", type=int, default=60)
    args = parser.parse_args()

    reference_rows = load_rows(args.reference, args.map_size, parse_set(args.reference_teams) or None)
    candidate_rows = load_rows(args.candidate, args.map_size, parse_set(args.candidate_teams) or None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(f"reference: {args.reference}\n")
        out.write(f"candidate: {args.candidate}\n")
        out.write(f"map_size: {args.map_size}\n")
        out.write(f"reference_rows: {len(reference_rows)}\n")
        out.write(f"candidate_rows: {len(candidate_rows)}\n")

        ref_turn = summarize(reference_rows, lambda row: turn_bucket(as_int(row, "turn")), DEFAULT_METRICS)
        cand_turn = summarize(candidate_rows, lambda row: turn_bucket(as_int(row, "turn")), DEFAULT_METRICS)
        write_summary_table(out, "reference_by_turn_bucket", ref_turn, DEFAULT_METRICS)
        write_summary_table(out, "candidate_by_turn_bucket", cand_turn, DEFAULT_METRICS)

        out.write("\n[top_candidate_minus_reference_diffs]\n")
        out.write("label,metric,reference,candidate,diff,reference_n,candidate_n\n")
        for item in diff_table(ref_turn, cand_turn, DEFAULT_METRICS, args.min_rows)[: args.top_diffs]:
            out.write(
                f"{item['label']},{item['metric']},{item['reference']:.4f},"
                f"{item['candidate']:.4f},{item['diff']:.4f},"
                f"{item['reference_n']},{item['candidate_n']}\n"
            )

        for prefix, rows in [("reference", reference_rows), ("candidate", candidate_rows)]:
            for name, table in loss_correlation_tables(rows, args.min_rows, DEFAULT_METRICS).items():
                write_summary_table(out, f"{prefix}_loss_by_{name}", table, DEFAULT_METRICS)

    print(f"reference rows: {len(reference_rows)}")
    print(f"candidate rows: {len(candidate_rows)}")
    print(f"analysis: {args.output}")


if __name__ == "__main__":
    main()
