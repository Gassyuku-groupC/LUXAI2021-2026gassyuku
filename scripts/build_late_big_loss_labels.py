#!/usr/bin/env python3
"""Build team-turn labels for late-game big city loss warnings.

This label layer is intentionally macro-oriented. Unlike fuel support labels,
it does not describe a concrete worker action. It asks whether a team state
after a configurable start turn is likely to suffer a large city-tile loss in
the next horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_strategy_features import iter_replay_paths, load_replay_bundle, rows_for_bundle  # noqa: E402


DEFAULT_PATTERNS = [
    "outputs/diagnostic_layer/best_agent_public_opponents_v1_16/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_public_opponents_v2_16/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_opponent_pool_v1/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_v10_random2_16_allopponents/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_seed1259068876_16_baseline/replays/map_16x16_vs_*.json",
]


BASE_FIELDS = [
    "label_type",
    "label_version",
    "file",
    "source_format",
    "submission_id",
    "episode_id",
    "source_opponent",
    "eval_side",
    "team",
    "team_name",
    "opponent_name",
    "rank",
    "reward",
    "map_size",
    "width",
    "height",
    "turn",
    "turns_remaining",
    "night_cycle",
    "cycle_turn",
    "phase",
    "pre_night",
    "is_night",
    "turns_to_night",
]

FEATURE_FIELDS = [
    "cities",
    "city_tiles",
    "largest_city_size",
    "mean_city_size",
    "resource_near_cities",
    "isolated_cities_r3",
    "units",
    "workers",
    "carts",
    "unit_cap_margin",
    "worker_citytile_ratio",
    "research",
    "fuel",
    "upkeep",
    "fuel_turns_total",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "median_city_fuel_turns",
    "mean_city_fuel_turns",
    "low_fuel_city_lt3",
    "low_fuel_city_lt5",
    "low_fuel_city_lt10",
    "unit_cargo_fuel",
    "wood_remaining",
    "coal_remaining",
    "uranium_remaining",
    "action_count",
    "move_actions",
    "transfer_actions",
    "pillage_actions",
    "research_actions",
    "bw_actions",
    "bc_actions",
    "bcity_actions",
    "bcity_isolated_actions",
    "bcity_adjacent_actions",
    "bcity_resource_near_actions",
    "bcity_adjacent_low_fuel_lt5_actions",
    "bw_low_fuel_lt3_actions",
    "bw_low_fuel_lt5_actions",
    "bw_low_fuel_lt10_actions",
]

TREND_FIELDS = [
    "city_tiles_delta_10",
    "city_tiles_growth_10",
    "workers_delta_10",
    "upkeep_delta_10",
    "upkeep_growth_10",
    "fuel_delta_10",
    "fuel_turns_total_delta_10",
    "fuel_turns_total_drop_10",
    "p25_city_fuel_turns_delta_10",
    "p25_city_fuel_turns_drop_10",
    "min_city_fuel_turns_delta_10",
    "research_delta_10",
    "research_growth_10",
]

LABEL_FIELDS = [
    "future_team_loss_5",
    "future_team_loss_10",
    "future_team_loss_20",
    "future_team_loss_30",
    "future_big_loss_horizon",
    "big_loss_threshold",
    "late_big_loss_warning",
    "late_big_loss_weight",
    "final_city_tiles",
    "final_units",
    "city_tiles_delta_next",
    "units_delta_next",
    "research_delta_next",
]

FIELDNAMES = BASE_FIELDS + FEATURE_FIELDS + TREND_FIELDS + LABEL_FIELDS


def as_float(row: dict, key: str) -> float:
    return float(row.get(key, 0) or 0)


def as_int(row: dict, key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def source_opponent_from_file(path: str) -> str:
    stem = Path(path).stem
    match = re.match(r"map_\d+x\d+_vs_(?P<opponent>.+)_\d+_p[01]$", stem)
    return match.group("opponent") if match else ""


def eval_side_from_file(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_p0"):
        return "0"
    if stem.endswith("_p1"):
        return "1"
    return ""


def positive_weight(row: dict, threshold: int, horizon: int) -> float:
    loss = as_float(row, f"future_team_loss_{horizon}")
    weight = 1.0
    if loss >= threshold * 2:
        weight += 0.75
    elif loss >= threshold * 1.5:
        weight += 0.35
    if as_float(row, "p25_city_fuel_turns") < 10:
        weight += 0.25
    if as_float(row, "fuel_turns_total_drop_10") > 10:
        weight += 0.25
    if as_float(row, "city_tiles_growth_10") > 0 and as_float(row, "fuel_turns_total_drop_10") > 5:
        weight += 0.25
    return round(weight, 4)


def add_trends(rows: list[dict], trend_window: int) -> list[dict]:
    by_key: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("file", "")), as_int(row, "team")), []).append(row)

    enriched = []
    for group in by_key.values():
        group.sort(key=lambda item: as_int(item, "turn"))
        by_turn = {as_int(row, "turn"): row for row in group}
        for row in group:
            turn = as_int(row, "turn")
            prev = by_turn.get(turn - trend_window)
            if prev is None:
                prev = group[0]

            city_delta = as_float(row, "city_tiles") - as_float(prev, "city_tiles")
            upkeep_delta = as_float(row, "upkeep") - as_float(prev, "upkeep")
            fuel_delta = as_float(row, "fuel") - as_float(prev, "fuel")
            fuel_turns_delta = as_float(row, "fuel_turns_total") - as_float(prev, "fuel_turns_total")
            p25_delta = as_float(row, "p25_city_fuel_turns") - as_float(prev, "p25_city_fuel_turns")
            min_delta = as_float(row, "min_city_fuel_turns") - as_float(prev, "min_city_fuel_turns")
            research_delta = as_float(row, "research") - as_float(prev, "research")

            row["city_tiles_delta_10"] = round(city_delta, 4)
            row["city_tiles_growth_10"] = round(max(city_delta, 0.0), 4)
            row["workers_delta_10"] = round(as_float(row, "workers") - as_float(prev, "workers"), 4)
            row["upkeep_delta_10"] = round(upkeep_delta, 4)
            row["upkeep_growth_10"] = round(max(upkeep_delta, 0.0), 4)
            row["fuel_delta_10"] = round(fuel_delta, 4)
            row["fuel_turns_total_delta_10"] = round(fuel_turns_delta, 4)
            row["fuel_turns_total_drop_10"] = round(max(-fuel_turns_delta, 0.0), 4)
            row["p25_city_fuel_turns_delta_10"] = round(p25_delta, 4)
            row["p25_city_fuel_turns_drop_10"] = round(max(-p25_delta, 0.0), 4)
            row["min_city_fuel_turns_delta_10"] = round(min_delta, 4)
            row["research_delta_10"] = round(research_delta, 4)
            row["research_growth_10"] = round(max(research_delta, 0.0), 4)
            enriched.append(row)
    return enriched


def make_label(row: dict, args: argparse.Namespace) -> dict | None:
    turn = as_int(row, "turn")
    if turn < args.start_turn:
        return None
    if args.max_turn and turn > args.max_turn:
        return None
    if as_int(row, "city_tiles") <= 0:
        return None

    future_loss = as_float(row, f"future_team_loss_{args.horizon}")
    label = int(future_loss >= args.big_loss_threshold)
    out = {field: row.get(field, "") for field in FIELDNAMES}
    out["label_type"] = "late_big_loss_warning"
    out["label_version"] = args.label_version
    out["source_opponent"] = source_opponent_from_file(str(row.get("file", "")))
    out["eval_side"] = eval_side_from_file(str(row.get("file", "")))
    out["night_cycle"] = turn // 40
    out["future_big_loss_horizon"] = args.horizon
    out["big_loss_threshold"] = args.big_loss_threshold
    out["late_big_loss_warning"] = label
    out["late_big_loss_weight"] = positive_weight(row, args.big_loss_threshold, args.horizon) if label else 1.0
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, labels: list[dict], replay_count: int, skipped_count: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "label_type": "late_big_loss_warning",
        "label_version": args.label_version,
        "replays": replay_count,
        "skipped_replays": skipped_count,
        "rows": len(labels),
        "start_turn": args.start_turn,
        "max_turn": args.max_turn,
        "horizon": args.horizon,
        "big_loss_threshold": args.big_loss_threshold,
    }
    if labels:
        data = pd.DataFrame(labels)
        data["late_big_loss_warning"] = pd.to_numeric(data["late_big_loss_warning"], errors="coerce").fillna(0)
        loss_key = f"future_team_loss_{args.horizon}"
        data[loss_key] = pd.to_numeric(data[loss_key], errors="coerce").fillna(0)
        data["turn"] = pd.to_numeric(data["turn"], errors="coerce").fillna(0)
        data["team"] = pd.to_numeric(data["team"], errors="coerce").fillna(-1)
        summary.update(
            {
                "positive_rows": int(data["late_big_loss_warning"].sum()),
                "positive_rate": float(data["late_big_loss_warning"].mean()),
                f"mean_future_loss_{args.horizon}": float(data[loss_key].mean()),
                f"max_future_loss_{args.horizon}": float(data[loss_key].max()),
                "by_night_cycle": data.groupby("night_cycle")["late_big_loss_warning"].agg(["count", "sum", "mean"]).to_dict("index"),
                "by_team": data.groupby("team")["late_big_loss_warning"].agg(["count", "sum", "mean"]).to_dict("index"),
            }
        )
        if "source_opponent" in data:
            summary["by_source_opponent"] = (
                data.groupby("source_opponent")["late_big_loss_warning"].agg(["count", "sum", "mean"]).to_dict("index")
            )
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_patterns(values: list[str]) -> list[str]:
    if values:
        return values
    return DEFAULT_PATTERNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Build late-game big city loss warning labels from Lux replays.")
    parser.add_argument("patterns", nargs="*", help="Replay glob patterns. Defaults to current best-agent diagnostic pools.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/late_big_loss_warning_v1"))
    parser.add_argument("--label-version", default="late_big_loss_warning_v1")
    parser.add_argument("--map-sizes", default="16", help="Comma-separated map sizes to keep. Empty keeps all.")
    parser.add_argument("--team-names", default="", help="Comma-separated team names to keep. Empty keeps both teams.")
    parser.add_argument("--start-turn", type=int, default=160)
    parser.add_argument("--max-turn", type=int, default=360)
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 10, 20, 30])
    parser.add_argument("--big-loss-threshold", type=int, default=10)
    parser.add_argument("--trend-window", type=int, default=10)
    parser.add_argument("--max-replays", type=int, default=0)
    args = parser.parse_args()

    horizons = sorted({5, 10, args.horizon, 30})
    map_filter = {int(part.strip()) for part in args.map_sizes.split(",") if part.strip()}
    team_filter = {part.strip() for part in args.team_names.split(",") if part.strip()}
    labels = []
    replay_count = 0
    skipped_count = 0

    for path in iter_replay_paths(parse_patterns(args.patterns)):
        bundle = load_replay_bundle(path)
        if bundle is None:
            skipped_count += 1
            continue
        if map_filter and int(bundle["width"]) not in map_filter:
            continue
        if team_filter and not any(name in team_filter for name in bundle["team_names"]):
            continue
        rows = rows_for_bundle(bundle, horizons)
        rows = add_trends(rows, args.trend_window)
        if team_filter:
            rows = [row for row in rows if row.get("team_name", "") in team_filter]
        for row in rows:
            label = make_label(row, args)
            if label is not None:
                labels.append(label)
        replay_count += 1
        if args.max_replays and replay_count >= args.max_replays:
            break

    output_csv = args.output_dir / "late_big_loss_labels.csv"
    summary_json = args.output_dir / "late_big_loss_summary.json"
    write_csv(output_csv, labels)
    write_summary(summary_json, labels, replay_count, skipped_count, args)
    print(f"replays: {replay_count}")
    print(f"skipped: {skipped_count}")
    print(f"rows: {len(labels)}")
    if labels:
        positive = sum(int(row["late_big_loss_warning"]) for row in labels)
        print(f"positive rows: {positive}")
        print(f"positive rate: {positive / max(len(labels), 1):.4f}")
    print(f"labels: {output_csv}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
