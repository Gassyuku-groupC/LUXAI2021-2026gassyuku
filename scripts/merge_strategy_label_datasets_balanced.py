#!/usr/bin/env python3
"""Merge strategy-label CSVs with optional per-map-size balancing.

This is meant for training all-map tabular scorers without letting the much
larger 16x16 replay pool dominate 12/24/32 behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_csv_ints(text: str) -> set[int]:
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            print(f"skip missing: {path}")
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["_source_csv"] = str(path)
        frames.append(frame)
        print(f"loaded {len(frame)} rows: {path}")
    if not frames:
        raise ValueError("No input CSVs were loaded.")
    data = pd.concat(frames, ignore_index=True)
    if "map_size" not in data.columns:
        raise ValueError("Input CSVs must contain map_size.")
    data["map_size"] = pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int)
    return data


def downsample_by_map_size(data: pd.DataFrame, per_map_max: int, seed: int) -> pd.DataFrame:
    if per_map_max <= 0:
        return data
    parts = []
    for map_size, group in data.groupby("map_size", sort=True):
        if len(group) > per_map_max:
            group = group.sample(n=per_map_max, random_state=seed)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def write_summary(path: Path, data: pd.DataFrame, input_paths: list[Path], args: argparse.Namespace) -> None:
    summary = {
        "input_csvs": [str(path) for path in input_paths],
        "output_csv": str(args.output_csv),
        "rows": int(len(data)),
        "map_sizes": args.map_sizes,
        "per_map_max_rows": args.per_map_max_rows,
        "source_csv_count": int(data["_source_csv"].nunique()) if "_source_csv" in data.columns else 0,
        "by_map_size": data["map_size"].value_counts().sort_index().to_dict(),
    }
    if "strategy_label" in data.columns:
        summary["strategy_label_counts"] = data["strategy_label"].value_counts().to_dict()
    for column in [
        "risk_city_loss_20",
        "risk_big_loss_20",
        "error_failed_with_big_loss",
        "expansion_success_40",
        "success_win",
    ]:
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce").fillna(0)
            summary[f"{column}_rate"] = float(values.mean())
    if "source_opponent" in data.columns:
        summary["by_source_opponent"] = data["source_opponent"].value_counts().head(50).to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and balance Lux strategy-label datasets.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input strategy_label_dataset*.csv files.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--map-sizes", default="", help="Comma-separated map sizes to keep. Empty keeps all.")
    parser.add_argument("--per-map-max-rows", type=int, default=0, help="Downsample each map size to this many rows.")
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    data = load_inputs(args.inputs)
    map_filter = parse_csv_ints(args.map_sizes)
    if map_filter:
        data = data[data["map_size"].isin(map_filter)].copy()
    if data.empty:
        raise ValueError("No rows remain after map-size filtering.")

    data = data.drop_duplicates(subset=[column for column in ["file", "team", "turn"] if column in data.columns])
    data = downsample_by_map_size(data, args.per_map_max_rows, args.seed)
    data = data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_csv, index=False, encoding="utf-8")
    write_summary(args.summary_json, data, args.inputs, args)
    print(f"rows: {len(data)}")
    print(f"by_map_size: {data['map_size'].value_counts().sort_index().to_dict()}")
    print(f"output: {args.output_csv}")
    print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
