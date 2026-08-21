#!/usr/bin/env python3
"""Build side/phase failure modifiers from strategy-feature CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def as_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" else int(float(value))


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


def side_from_file(path: str) -> str:
    match = re.search(r"_p([01])\.json$", path.replace("\\", "/"))
    if match:
        return f"p{match.group(1)}"
    return "unknown"


def load_rows(paths: list[Path], map_size: int) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as in_file:
            for row in csv.DictReader(in_file):
                if map_size and as_int(row, "map_size") != map_size:
                    continue
                row["_side"] = side_from_file(row.get("file", ""))
                row["_bucket"] = turn_bucket(as_int(row, "turn"))
                rows.append(row)
    return rows


def summarize(rows: list[dict], min_rows: int, loss_threshold: float) -> tuple[list[dict], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["_side"] == "unknown":
            continue
        groups[(row["_side"], row["_bucket"])].append(row)

    summaries = []
    modifiers: dict[str, dict] = {"side_phase": {}}
    for (side, bucket), group in sorted(groups.items()):
        if len(group) < min_rows:
            continue
        n = len(group)
        future_loss = sum(as_float(row, "future_team_loss_10") for row in group) / n
        loss_rate = sum(1.0 for row in group if as_float(row, "future_team_loss_10") > 0) / n
        big_loss_rate = sum(1.0 for row in group if as_float(row, "future_team_loss_10") >= loss_threshold) / n
        min_buffer = sum(as_float(row, "min_city_fuel_turns") for row in group) / n
        p25_buffer = sum(as_float(row, "p25_city_fuel_turns") for row in group) / n
        city_tiles = sum(as_float(row, "city_tiles") for row in group) / n
        final_city_tiles = sum(as_float(row, "final_city_tiles") for row in group) / n
        bw_low_fuel = sum(as_float(row, "bw_low_fuel_lt5_actions") for row in group) / n
        bcity_low_fuel = sum(as_float(row, "bcity_adjacent_low_fuel_lt5_actions") for row in group) / n

        severity = 0
        if big_loss_rate >= 0.10 or future_loss >= loss_threshold:
            severity = 3
        elif loss_rate >= 0.25 or future_loss >= 1.0:
            severity = 2
        elif loss_rate >= 0.10 or future_loss >= 0.4:
            severity = 1

        item = {
            "side": side,
            "bucket": bucket,
            "n": n,
            "future_loss_10": round(future_loss, 4),
            "loss_rate_10": round(loss_rate, 4),
            "big_loss_rate_10": round(big_loss_rate, 4),
            "min_city_fuel_turns": round(min_buffer, 4),
            "p25_city_fuel_turns": round(p25_buffer, 4),
            "city_tiles": round(city_tiles, 4),
            "final_city_tiles": round(final_city_tiles, 4),
            "bw_low_fuel_lt5_actions": round(bw_low_fuel, 4),
            "bcity_adjacent_low_fuel_lt5_actions": round(bcity_low_fuel, 4),
            "severity": severity,
        }
        summaries.append(item)
        if severity > 0:
            modifiers["side_phase"][f"{side}:{bucket}"] = {
                "severity": severity,
                "future_loss_10": item["future_loss_10"],
                "loss_rate_10": item["loss_rate_10"],
                "big_loss_rate_10": item["big_loss_rate_10"],
            }

    return summaries, modifiers


def write_report(path: Path, summaries: list[dict], modifiers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "side,phase,n,future_loss_10,loss_rate_10,big_loss_rate_10,min_buffer,p25_buffer,city_tiles,final_city_tiles,bw_low_fuel_lt5,bcity_adj_low_fuel_lt5,severity"
    ]
    for row in summaries:
        lines.append(
            ",".join(
                str(row[key])
                for key in [
                    "side",
                    "bucket",
                    "n",
                    "future_loss_10",
                    "loss_rate_10",
                    "big_loss_rate_10",
                    "min_city_fuel_turns",
                    "p25_city_fuel_turns",
                    "city_tiles",
                    "final_city_tiles",
                    "bw_low_fuel_lt5_actions",
                    "bcity_adjacent_low_fuel_lt5_actions",
                    "severity",
                ]
            )
        )
    lines.append("")
    lines.append("[modifiers]")
    lines.append(json.dumps(modifiers, ensure_ascii=False, indent=2))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze side/phase failures from strategy features.")
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/risk_feature_logs/side_phase_failure_report.txt"))
    parser.add_argument("--modifiers-output", type=Path, default=Path("outputs/risk_feature_logs/side_phase_failure_modifiers.json"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--big-loss-threshold", type=float, default=5.0)
    args = parser.parse_args()

    rows = load_rows(args.features, args.map_size)
    summaries, modifiers = summarize(rows, args.min_rows, args.big_loss_threshold)
    write_report(args.output, summaries, modifiers)
    args.modifiers_output.parent.mkdir(parents=True, exist_ok=True)
    args.modifiers_output.write_text(json.dumps(modifiers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rows: {len(rows)}")
    print(f"groups: {len(summaries)}")
    print(f"active modifiers: {len(modifiers['side_phase'])}")
    print(f"report: {args.output}")
    print(f"modifiers: {args.modifiers_output}")


if __name__ == "__main__":
    main()
