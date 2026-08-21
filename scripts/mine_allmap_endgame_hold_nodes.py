#!/usr/bin/env python3
"""Mine expansion-to-hold transition nodes for each Lux map size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "map_size",
    "turn",
    "city_tiles",
    "p25_city_fuel_turns",
    "min_city_fuel_turns",
    "bw_actions",
    "bcity_actions",
    "city_tiles_growth_10",
    "future_team_loss_20",
    "future_city_tiles_gain_40",
    "expansion_success_40",
    "success_win",
]


def load_data(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["_source_csv"] = str(path)
        frames.append(frame)
        print(f"loaded {len(frame)} rows: {path}")
    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            data[column] = 0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["map_size"] = data["map_size"].astype(int)
    data["turn_bin"] = (data["turn"] // 20 * 20).astype(int)
    data["expansion_pressure"] = (
        (data["bw_actions"] > 0)
        | (data["bcity_actions"] > 0)
        | (data["city_tiles_growth_10"] > 0)
    ).astype(int)
    data["future_big_loss"] = (data["future_team_loss_20"] >= 10).astype(int)
    data["future_loss_any"] = (data["future_team_loss_20"] > 0).astype(int)
    data["future_low_gain"] = (data["future_city_tiles_gain_40"] <= 2).astype(int)
    return data


def summarize(group: pd.DataFrame) -> dict:
    return {
        "rows": int(len(group)),
        "replays": int(group["file"].nunique()) if "file" in group.columns else 0,
        "win_rate": float(group["success_win"].mean()),
        "mean_city_tiles": float(group["city_tiles"].mean()),
        "median_city_tiles": float(group["city_tiles"].median()),
        "p25_fuel_turns": float(group["p25_city_fuel_turns"].median()),
        "expansion_pressure_rate": float(group["expansion_pressure"].mean()),
        "expansion_success_rate": float(group["expansion_success_40"].mean()),
        "future_loss20_rate": float(group["future_loss_any"].mean()),
        "future_big_loss20_rate": float(group["future_big_loss"].mean()),
        "future_gain40_mean": float(group["future_city_tiles_gain_40"].mean()),
        "low_gain40_rate": float(group["future_low_gain"].mean()),
    }


def infer_hold_turn(turn_bins: list[dict]) -> int:
    candidates = [
        row for row in turn_bins
        if row["turn_bin"] >= 200
        and row["low_gain40_rate"] >= 0.50
        and row["future_gain40_mean"] <= max(10.0, row["mean_city_tiles"] * 0.08)
    ]
    if candidates:
        return int(candidates[0]["turn_bin"])
    late = [row for row in turn_bins if row["turn_bin"] >= 300]
    if late:
        return int(max(late, key=lambda row: row["low_gain40_rate"])["turn_bin"])
    return 340


def current_state_rules(data: pd.DataFrame, min_rows: int) -> list[dict]:
    rows = []
    for turn in [180, 200, 220, 240, 260, 280, 300, 320, 340]:
        for city in [10, 20, 30, 40, 60, 80, 100, 120, 140]:
            for fuel in [5, 8, 10, 12, 15, 20]:
                group = data[
                    (data["turn"] >= turn)
                    & (data["city_tiles"] >= city)
                    & (data["p25_city_fuel_turns"] >= fuel)
                ]
                if len(group) < min_rows:
                    continue
                expansion = group[group["expansion_pressure"] > 0]
                no_expansion = group[group["expansion_pressure"] == 0]
                if len(expansion) < max(100, min_rows // 10) or len(no_expansion) < max(50, min_rows // 20):
                    continue
                item = summarize(group)
                item.update(
                    {
                        "rule": f"turn>={turn}, city_tiles>={city}, p25_fuel>={fuel}",
                        "turn_threshold": turn,
                        "city_tiles_threshold": city,
                        "p25_fuel_threshold": fuel,
                        "expansion_big_loss20_rate": float(expansion["future_big_loss"].mean()),
                        "no_expansion_big_loss20_rate": float(no_expansion["future_big_loss"].mean()),
                        "expansion_gain40_mean": float(expansion["future_city_tiles_gain_40"].mean()),
                        "no_expansion_gain40_mean": float(no_expansion["future_city_tiles_gain_40"].mean()),
                    }
                )
                item["hold_score"] = (
                    item["low_gain40_rate"]
                    - item["expansion_success_rate"]
                    + item["future_big_loss20_rate"]
                    + max(0.0, item["expansion_big_loss20_rate"] - item["no_expansion_big_loss20_rate"])
                )
                rows.append(item)
    return sorted(rows, key=lambda row: (row["hold_score"], row["rows"]), reverse=True)


def map_summary(data: pd.DataFrame, map_size: int, min_rows: int) -> dict:
    subset = data[data["map_size"] == map_size].copy()
    turn_bins = []
    for turn_bin, group in subset.groupby("turn_bin", sort=True):
        item = summarize(group)
        item["turn_bin"] = int(turn_bin)
        turn_bins.append(item)
    rules = current_state_rules(subset, min_rows)
    return {
        "map_size": map_size,
        "overall": summarize(subset),
        "inferred_hold_turn": infer_hold_turn(turn_bins),
        "turn_bins": turn_bins,
        "top_current_state_rules": rules[:30],
    }


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# All-Map Endgame Hold Node Mining",
        "",
        f"- inputs: `{', '.join(summary['input_csvs'])}`",
        "",
    ]
    for item in summary["maps"]:
        lines.extend(
            [
                f"## {item['map_size']}x{item['map_size']}",
                "",
                f"- rows: {item['overall']['rows']}",
                f"- replays: {item['overall']['replays']}",
                f"- inferred_hold_turn: {item['inferred_hold_turn']}",
                "",
                "| turn | rows | city | p25 fuel | expand | exp ok | big20 | gain40 | low gain |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["turn_bins"]:
            lines.append(
                f"| {row['turn_bin']} | {row['rows']} | {row['mean_city_tiles']:.1f} | "
                f"{row['p25_fuel_turns']:.1f} | {row['expansion_pressure_rate']:.3f} | "
                f"{row['expansion_success_rate']:.3f} | {row['future_big_loss20_rate']:.3f} | "
                f"{row['future_gain40_mean']:.2f} | {row['low_gain40_rate']:.3f} |"
            )
        lines.extend(
            [
                "",
                "| top rule | rows | expand | exp ok | big20 | exp big20 | no-exp big20 | gain40 | low gain | score |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["top_current_state_rules"][:10]:
            lines.append(
                f"| {row['rule']} | {row['rows']} | {row['expansion_pressure_rate']:.3f} | "
                f"{row['expansion_success_rate']:.3f} | {row['future_big_loss20_rate']:.3f} | "
                f"{row['expansion_big_loss20_rate']:.3f} | {row['no_expansion_big_loss20_rate']:.3f} | "
                f"{row['future_gain40_mean']:.2f} | {row['low_gain40_rate']:.3f} | {row['hold_score']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine all-map endgame hold nodes.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/diagnostic_layer/mine_allmap_endgame_hold_nodes_v1/summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/diagnostic_layer/mine_allmap_endgame_hold_nodes_v1/report.md"))
    parser.add_argument("--map-sizes", default="12,16,24,32")
    parser.add_argument("--min-rule-rows", type=int, default=1500)
    args = parser.parse_args()

    data = load_data(args.inputs)
    map_sizes = [int(part.strip()) for part in args.map_sizes.split(",") if part.strip()]
    summary = {
        "input_csvs": [str(path) for path in args.inputs],
        "maps": [map_summary(data, map_size, args.min_rule_rows) for map_size in map_sizes],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output_md, summary)
    for item in summary["maps"]:
        print(f"{item['map_size']}x{item['map_size']}: hold_turn={item['inferred_hold_turn']} rows={item['overall']['rows']}")
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_md}")


if __name__ == "__main__":
    main()
