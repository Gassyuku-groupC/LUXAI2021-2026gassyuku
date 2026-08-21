#!/usr/bin/env python3
"""Summarize action-level risk feature correlations."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


def as_int(row: dict, key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def is_true(row: dict, key: str) -> bool:
    return as_int(row, key) != 0


def summarize(rows: list[dict], label: str, horizon: int) -> dict:
    key = f"future_team_loss_{horizon}"
    losses = [as_float(row, key, 0.0) for row in rows]
    if not losses:
        return {"label": label, "n": 0, "loss_rate": 0.0, "big_loss_rate": 0.0, "mean_loss": 0.0}
    return {
        "label": label,
        "n": len(losses),
        "loss_rate": sum(loss > 0 for loss in losses) / len(losses),
        "big_loss_rate": sum(loss >= 5 for loss in losses) / len(losses),
        "mean_loss": sum(losses) / len(losses),
    }


def bucket_rows(rows: list[dict], label_fn: Callable[[dict], str], horizon: int) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[label_fn(row)].append(row)
    return [summarize(bucket, label, horizon) for label, bucket in sorted(buckets.items())]


def print_table(title: str, summaries: Iterable[dict], out_file) -> None:
    line = f"\n[{title}]"
    print(line)
    out_file.write(line + "\n")
    header = "label,n,loss_rate,big_loss_rate,mean_loss"
    print(header)
    out_file.write(header + "\n")
    for item in summaries:
        line = (
            f"{item['label']},{item['n']},"
            f"{item['loss_rate']:.4f},{item['big_loss_rate']:.4f},{item['mean_loss']:.4f}"
        )
        print(line)
        out_file.write(line + "\n")


def phase(row: dict) -> str:
    if is_true(row, "is_night"):
        return "night"
    if is_true(row, "pre_night"):
        return "pre_night"
    return "day"


def fuel_bucket(row: dict, key: str) -> str:
    fuel = as_float(row, key)
    if math.isnan(fuel):
        return f"{key}=missing"
    if fuel < 3:
        return f"{key}<3"
    if fuel < 5:
        return f"{key}=3-5"
    if fuel < 10:
        return f"{key}=5-10"
    return f"{key}>=10"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze action-level risk features.")
    parser.add_argument("--input", type=Path, default=Path("outputs/risk_feature_logs/action_risk_features_outputs_all.csv"))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("outputs/risk_feature_logs/action_risk_summary_16x16.txt"))
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as in_file:
        rows = list(csv.DictReader(in_file))
    if args.map_size:
        rows = [row for row in rows if as_int(row, "map_size") == args.map_size]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_file:
        header = f"input: {args.input}\nrows: {len(rows)}\nhorizon: {args.horizon}\nmap_size: {args.map_size}\n"
        print(header, end="")
        out_file.write(header)

        print_table(
            "action_phase",
            bucket_rows(rows, lambda row: f"{row['action']}|{phase(row)}", args.horizon),
            out_file,
        )
        print_table(
            "action_phase_target_fuel",
            [
                item
                for item in bucket_rows(
                    [row for row in rows if row["action"] in ("bw", "bc")],
                    lambda row: f"{row['action']}|{phase(row)}|{fuel_bucket(row, 'target_city_fuel_turns')}",
                    args.horizon,
                )
                if item["n"] >= args.min_rows
            ],
            out_file,
        )
        print_table(
            "bcity_context",
            [
                item
                for item in bucket_rows(
                    [row for row in rows if row["action"] == "bcity"],
                    lambda row: (
                        f"bcity|{phase(row)}|near={as_int(row, 'near_resource')}|"
                        f"isolated={as_int(row, 'isolated')}|"
                        f"{fuel_bucket(row, 'min_adjacent_city_fuel_turns')}"
                    ),
                    args.horizon,
                )
                if item["n"] >= args.min_rows
            ],
            out_file,
        )
        print_table(
            "top_high_risk_rules",
            sorted(
                [
                    item
                    for item in bucket_rows(
                        rows,
                        lambda row: (
                            f"{row['action']}|{phase(row)}|"
                            f"target={fuel_bucket(row, 'target_city_fuel_turns')}|"
                            f"adj={fuel_bucket(row, 'min_adjacent_city_fuel_turns')}"
                        ),
                        args.horizon,
                    )
                    if item["n"] >= args.min_rows
                ],
                key=lambda item: (item["loss_rate"], item["big_loss_rate"], item["n"]),
                reverse=True,
            )[:40],
            out_file,
        )
    print(f"\nsummary: {args.output}")


if __name__ == "__main__":
    main()
