#!/usr/bin/env python3
"""Mine 32x32 replay states for expansion-to-hold transition nodes.

The output is diagnostic: it looks for turn/score/fuel regions where expansion
actions stop adding reliable final scale and start correlating with future city
loss. This can guide a late-game gate without hard-coding it blindly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "map_size",
    "turn",
    "turns_remaining",
    "city_tiles",
    "workers",
    "units",
    "unit_cap_margin",
    "worker_citytile_ratio",
    "research",
    "fuel_turns_total",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "mean_city_fuel_turns",
    "low_fuel_city_lt5",
    "bw_actions",
    "bcity_actions",
    "city_tiles_growth_10",
    "workers_growth_10",
    "fuel_turns_total_drop_10",
    "future_team_loss_10",
    "future_team_loss_20",
    "future_team_loss_30",
    "future_city_tiles_gain_20",
    "future_city_tiles_gain_40",
    "final_city_tiles",
    "final_opponent_city_tiles",
    "final_city_tile_margin",
    "success_win",
    "risk_big_loss_20",
    "expansion_taken",
    "expansion_success_40",
]


def read_data(path: Path) -> pd.DataFrame:
    usecols = None
    data = pd.read_csv(path, usecols=usecols, low_memory=False)
    data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == 32].copy()
    if data.empty:
        raise ValueError("No 32x32 rows found.")
    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            data[column] = 0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["turn_bin"] = (data["turn"] // 20 * 20).astype(int)
    data["city_lead_ratio"] = data["final_city_tile_margin"] / data[["final_city_tiles", "final_opponent_city_tiles"]].max(axis=1).clip(lower=1)
    data["state_city_ratio"] = data["city_tiles"] / data["final_opponent_city_tiles"].clip(lower=1)
    data["late_hold_candidate"] = (
        (data["turn"] >= 200)
        & (data["city_tiles"] >= 40)
        & (data["p25_city_fuel_turns"] >= 8)
        & (data["final_city_tile_margin"] >= 0)
    ).astype(int)
    data["expansion_pressure"] = ((data["bw_actions"] > 0) | (data["bcity_actions"] > 0) | (data["city_tiles_growth_10"] > 0)).astype(int)
    data["future_big_loss"] = (data["future_team_loss_20"] >= 10).astype(int)
    data["future_loss_any"] = (data["future_team_loss_20"] > 0).astype(int)
    data["future_low_gain"] = (data["future_city_tiles_gain_40"] <= 2).astype(int)
    return data


def summarize_group(group: pd.DataFrame) -> dict:
    return {
        "rows": int(len(group)),
        "replays": int(group["file"].nunique()) if "file" in group.columns else 0,
        "win_rate": float(group["success_win"].mean()),
        "mean_city_tiles": float(group["city_tiles"].mean()),
        "median_city_tiles": float(group["city_tiles"].median()),
        "mean_margin": float(group["final_city_tile_margin"].mean()),
        "p25_fuel_turns": float(group["p25_city_fuel_turns"].median()),
        "min_fuel_turns": float(group["min_city_fuel_turns"].median()),
        "expansion_pressure_rate": float(group["expansion_pressure"].mean()),
        "expansion_success_rate": float(group["expansion_success_40"].mean()),
        "future_loss20_rate": float(group["future_loss_any"].mean()),
        "future_big_loss20_rate": float(group["future_big_loss"].mean()),
        "future_gain40_mean": float(group["future_city_tiles_gain_40"].mean()),
        "future_gain40_median": float(group["future_city_tiles_gain_40"].median()),
        "low_gain40_rate": float(group["future_low_gain"].mean()),
    }


def turn_bin_summary(data: pd.DataFrame) -> list[dict]:
    rows = []
    for turn_bin, group in data.groupby("turn_bin", sort=True):
        item = summarize_group(group)
        item["turn_bin"] = int(turn_bin)
        rows.append(item)
    return rows


def candidate_rules(data: pd.DataFrame, min_rows: int) -> list[dict]:
    rows = []
    turn_thresholds = [160, 180, 200, 220, 240, 260, 280]
    city_thresholds = [30, 40, 60, 80, 100, 120]
    fuel_thresholds = [5, 8, 10, 15, 20]
    margin_thresholds = [-20, 0, 20, 50, 100]
    for turn in turn_thresholds:
        for city_tiles in city_thresholds:
            for fuel in fuel_thresholds:
                for margin in margin_thresholds:
                    mask = (
                        (data["turn"] >= turn)
                        & (data["city_tiles"] >= city_tiles)
                        & (data["p25_city_fuel_turns"] >= fuel)
                        & (data["final_city_tile_margin"] >= margin)
                    )
                    group = data[mask]
                    if len(group) < min_rows:
                        continue
                    expansion = group[group["expansion_pressure"] > 0]
                    no_expansion = group[group["expansion_pressure"] == 0]
                    if len(expansion) < max(50, min_rows // 10) or len(no_expansion) < max(50, min_rows // 10):
                        continue
                    item = summarize_group(group)
                    item.update(
                        {
                            "rule": f"turn>={turn}, city_tiles>={city_tiles}, p25_fuel>={fuel}, final_margin>={margin}",
                            "turn_threshold": turn,
                            "city_tiles_threshold": city_tiles,
                            "p25_fuel_threshold": fuel,
                            "final_margin_threshold": margin,
                            "expansion_rows": int(len(expansion)),
                            "no_expansion_rows": int(len(no_expansion)),
                            "expansion_big_loss20_rate": float(expansion["future_big_loss"].mean()),
                            "no_expansion_big_loss20_rate": float(no_expansion["future_big_loss"].mean()),
                            "expansion_gain40_mean": float(expansion["future_city_tiles_gain_40"].mean()),
                            "no_expansion_gain40_mean": float(no_expansion["future_city_tiles_gain_40"].mean()),
                            "expansion_success_rate_only": float(expansion["expansion_success_40"].mean()),
                            "no_expansion_win_rate": float(no_expansion["success_win"].mean()),
                        }
                    )
                    item["hold_score"] = (
                        item["win_rate"]
                        + item["low_gain40_rate"]
                        - item["expansion_success_rate"]
                        + max(0.0, item["expansion_big_loss20_rate"] - item["no_expansion_big_loss20_rate"])
                    )
                    rows.append(item)
    return sorted(rows, key=lambda item: (item["hold_score"], item["rows"]), reverse=True)


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# 32x32 Endgame Hold Node Mining",
        "",
        f"- input: `{summary['input_csv']}`",
        f"- rows_32: {summary['rows_32']}",
        f"- replays_32: {summary['replays_32']}",
        "",
        "## Turn Bins",
        "",
        "| turn | rows | win | city | margin | p25 fuel | expand | exp ok | loss20 | big20 | gain40 | low gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["turn_bins"]:
        lines.append(
            f"| {item['turn_bin']} | {item['rows']} | {item['win_rate']:.3f} | "
            f"{item['mean_city_tiles']:.1f} | {item['mean_margin']:.1f} | {item['p25_fuel_turns']:.1f} | "
            f"{item['expansion_pressure_rate']:.3f} | {item['expansion_success_rate']:.3f} | "
            f"{item['future_loss20_rate']:.3f} | {item['future_big_loss20_rate']:.3f} | "
            f"{item['future_gain40_mean']:.2f} | {item['low_gain40_rate']:.3f} |"
        )
    lines.extend(["", "## Top Hold Candidates", ""])
    lines.append("| rule | rows | win | expand | exp ok | exp big20 | no-exp big20 | exp gain40 | no-exp gain40 | hold score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in summary["top_rules"][:25]:
        lines.append(
            f"| {item['rule']} | {item['rows']} | {item['win_rate']:.3f} | "
            f"{item['expansion_pressure_rate']:.3f} | {item['expansion_success_rate']:.3f} | "
            f"{item['expansion_big_loss20_rate']:.3f} | {item['no_expansion_big_loss20_rate']:.3f} | "
            f"{item['expansion_gain40_mean']:.2f} | {item['no_expansion_gain40_mean']:.2f} | "
            f"{item['hold_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine 32x32 endgame hold nodes.")
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_non16_all_existing_v1.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/diagnostic_layer/mine_32_endgame_hold_nodes_v1/summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/diagnostic_layer/mine_32_endgame_hold_nodes_v1/report.md"))
    parser.add_argument("--min-rule-rows", type=int, default=1000)
    args = parser.parse_args()

    data = read_data(args.input_csv)
    summary = {
        "input_csv": str(args.input_csv),
        "rows_32": int(len(data)),
        "replays_32": int(data["file"].nunique()) if "file" in data.columns else 0,
        "overall": summarize_group(data),
        "turn_bins": turn_bin_summary(data),
        "top_rules": candidate_rules(data, args.min_rule_rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(f"rows_32: {summary['rows_32']}")
    print(f"replays_32: {summary['replays_32']}")
    print(f"rules: {len(summary['top_rules'])}")
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_md}")


if __name__ == "__main__":
    main()
