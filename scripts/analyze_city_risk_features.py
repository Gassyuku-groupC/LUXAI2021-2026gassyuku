#!/usr/bin/env python3
"""Summarize city-risk feature correlations."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


def as_int(row: dict, key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def as_float(row: dict, key: str) -> float:
    return float(row.get(key, 0) or 0)


def summarize(rows: list[dict], label: str, horizon: int) -> dict:
    key = f"future_team_loss_{horizon}"
    if not rows:
        return {
            "label": label,
            "n": 0,
            "loss_rate": 0.0,
            "mean_loss": 0.0,
            "big_loss_rate": 0.0,
        }
    losses = [as_float(row, key) for row in rows]
    return {
        "label": label,
        "n": len(rows),
        "loss_rate": sum(loss > 0 for loss in losses) / len(losses),
        "mean_loss": sum(losses) / len(losses),
        "big_loss_rate": sum(loss >= 5 for loss in losses) / len(losses),
    }


def print_table(title: str, summaries: Iterable[dict]) -> None:
    print(f"\n{title}")
    print("label,n,loss_rate,mean_loss,big_loss_rate")
    for item in summaries:
        print(
            f"{item['label']},{item['n']},"
            f"{item['loss_rate']:.4f},{item['mean_loss']:.4f},{item['big_loss_rate']:.4f}"
        )


def bucket_rows(rows: list[dict], label_fn: Callable[[dict], str], horizon: int) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[label_fn(row)].append(row)
    return [summarize(bucket_rows, label, horizon) for label, bucket_rows in sorted(buckets.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze city risk features.")
    parser.add_argument("--input", type=Path, default=Path("outputs/risk_feature_logs/city_risk_features_outputs_all.csv"))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--map-size", type=int, default=0)
    parser.add_argument("--min-turn", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/risk_feature_logs/city_risk_summary.txt"))
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as in_file:
        rows = list(csv.DictReader(in_file))
    if args.map_size:
        rows = [row for row in rows if as_int(row, "map_size") == args.map_size]
    if args.min_turn:
        rows = [row for row in rows if as_int(row, "turn") >= args.min_turn]

    def fuel_bucket(row: dict) -> str:
        fuel_turns = as_float(row, "fuel_turns")
        if fuel_turns < 3:
            return "fuel_turns <3"
        if fuel_turns < 6:
            return "fuel_turns 3-6"
        if fuel_turns < 10:
            return "fuel_turns 6-10"
        if fuel_turns < 15:
            return "fuel_turns 10-15"
        return "fuel_turns >=15"

    conditions = [
        ("all", lambda row: True),
        ("pre_or_night", lambda row: as_int(row, "pre_night") or as_int(row, "is_night")),
        ("night", lambda row: as_int(row, "is_night")),
        ("pre_night", lambda row: as_int(row, "pre_night")),
        ("upkeep_increased_next", lambda row: as_int(row, "upkeep_increased_next")),
        ("isolated_bcity", lambda row: as_int(row, "isolated_build_city_actions") > 0),
        ("resource_near_bcity", lambda row: as_int(row, "resource_near_build_city_actions") > 0),
        ("adjacent_bcity", lambda row: as_int(row, "adjacent_build_city_actions") > 0),
        ("big_city_low_fuel", lambda row: as_int(row, "city_size") >= 10 and as_float(row, "fuel_turns") < 10),
        (
            "big_city_low_fuel_pre_or_night",
            lambda row: as_int(row, "city_size") >= 10
            and as_float(row, "fuel_turns") < 10
            and (as_int(row, "pre_night") or as_int(row, "is_night")),
        ),
        (
            "low_fuel_upkeep_inc_pre_or_night",
            lambda row: as_float(row, "fuel_turns") < 10
            and as_int(row, "upkeep_increased_next")
            and (as_int(row, "pre_night") or as_int(row, "is_night")),
        ),
    ]

    lines = []
    print(f"input: {args.input}")
    print(f"rows: {len(rows)}")
    print(f"horizon: {args.horizon}")
    condition_summaries = [
        summarize([row for row in rows if predicate(row)], label, args.horizon)
        for label, predicate in conditions
    ]
    print_table("conditions", condition_summaries)
    print_table("fuel_buckets", bucket_rows(rows, fuel_bucket, args.horizon))
    print_table(
        "map_size",
        bucket_rows(rows, lambda row: f"map_{as_int(row, 'map_size')}", args.horizon),
    )
    print_table(
        "cycle_phase",
        bucket_rows(
            rows,
            lambda row: (
                "night"
                if as_int(row, "is_night")
                else "pre_night"
                if as_int(row, "pre_night")
                else "day"
            ),
            args.horizon,
        ),
    )

    # Save a compact CSV-like report too.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_file:
        out_file.write(f"input: {args.input}\nrows: {len(rows)}\nhorizon: {args.horizon}\n")
        for section, summaries in [
            ("conditions", condition_summaries),
            ("fuel_buckets", bucket_rows(rows, fuel_bucket, args.horizon)),
            ("map_size", bucket_rows(rows, lambda row: f"map_{as_int(row, 'map_size')}", args.horizon)),
            (
                "cycle_phase",
                bucket_rows(
                    rows,
                    lambda row: (
                        "night"
                        if as_int(row, "is_night")
                        else "pre_night"
                        if as_int(row, "pre_night")
                        else "day"
                    ),
                    args.horizon,
                ),
            ),
        ]:
            out_file.write(f"\n[{section}]\n")
            out_file.write("label,n,loss_rate,mean_loss,big_loss_rate\n")
            for item in summaries:
                out_file.write(
                    f"{item['label']},{item['n']},{item['loss_rate']:.4f},"
                    f"{item['mean_loss']:.4f},{item['big_loss_rate']:.4f}\n"
                )
    print(f"\nsummary: {args.output}")


if __name__ == "__main__":
    main()
