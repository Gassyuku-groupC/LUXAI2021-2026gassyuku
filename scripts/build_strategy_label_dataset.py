#!/usr/bin/env python3
"""Build a unified strategy-label dataset from Lux AI replay files.

This script is the offline "label layer" for replay review. It merges official
Kaggle replays and local evaluation replays into team-turn rows with labels for
risk, errors, expansion quality, and success. The output is meant for tabular
diagnostics first, not direct actor updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_strategy_features import (  # noqa: E402
    PLAYER_IDS,
    iter_replay_paths,
    load_replay_bundle,
    rows_for_bundle,
)


DEFAULT_PATTERNS = [
    "dataset/raw/data/**/*.json",
    "dataset/raw/**/*.json",
    "outputs/diagnostic_layer/**/replays/map_*_p[01].json",
]

BASE_FIELDS = [
    "label_version",
    "file",
    "source_kind",
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
    "win_label",
    "loss_label",
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
    "workers_growth_10",
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

FUTURE_FIELDS = [
    "future_team_loss_5",
    "future_team_loss_10",
    "future_team_loss_20",
    "future_team_loss_30",
    "future_team_loss_40",
    "future_city_tiles_gain_10",
    "future_city_tiles_gain_20",
    "future_city_tiles_gain_40",
    "future_workers_gain_20",
    "future_research_gain_20",
    "city_tiles_delta_next",
    "units_delta_next",
    "research_delta_next",
    "final_city_tiles",
    "final_units",
    "final_opponent_city_tiles",
    "final_opponent_units",
    "final_city_tile_margin",
    "final_unit_margin",
]

LABEL_FIELDS = [
    "risk_city_loss_10",
    "risk_city_loss_20",
    "risk_city_loss_30",
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
    "strategy_label",
    "sample_weight",
]

FIELDNAMES = BASE_FIELDS + FEATURE_FIELDS + TREND_FIELDS + FUTURE_FIELDS + LABEL_FIELDS

MANIFEST_FIELDS = [
    "file",
    "status",
    "reason",
    "source_kind",
    "source_format",
    "submission_id",
    "episode_id",
    "map_size",
    "width",
    "height",
    "team_0_name",
    "team_1_name",
    "rows_written",
]


def as_float(row: dict, key: str) -> float:
    return float(row.get(key, 0) or 0)


def as_int(row: dict, key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def parse_csv_ints(text: str) -> set[int]:
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def parse_csv_strings(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


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


def source_kind(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("outputs/") or "/outputs/" in normalized:
        return "local_eval"
    if normalized.startswith("dataset/raw/") or "/dataset/raw/" in normalized:
        return "official_or_downloaded"
    return "unknown"


def add_trends(rows: list[dict], trend_window: int) -> list[dict]:
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(str(row.get("file", "")), as_int(row, "team"))].append(row)

    enriched = []
    for group in by_key.values():
        group.sort(key=lambda item: as_int(item, "turn"))
        by_turn = {as_int(row, "turn"): row for row in group}
        for row in group:
            turn = as_int(row, "turn")
            prev = by_turn.get(turn - trend_window, group[0])
            city_delta = as_float(row, "city_tiles") - as_float(prev, "city_tiles")
            workers_delta = as_float(row, "workers") - as_float(prev, "workers")
            upkeep_delta = as_float(row, "upkeep") - as_float(prev, "upkeep")
            fuel_delta = as_float(row, "fuel") - as_float(prev, "fuel")
            fuel_turns_delta = as_float(row, "fuel_turns_total") - as_float(prev, "fuel_turns_total")
            p25_delta = as_float(row, "p25_city_fuel_turns") - as_float(prev, "p25_city_fuel_turns")
            min_delta = as_float(row, "min_city_fuel_turns") - as_float(prev, "min_city_fuel_turns")
            research_delta = as_float(row, "research") - as_float(prev, "research")

            row["city_tiles_delta_10"] = round(city_delta, 4)
            row["city_tiles_growth_10"] = round(max(city_delta, 0.0), 4)
            row["workers_delta_10"] = round(workers_delta, 4)
            row["workers_growth_10"] = round(max(workers_delta, 0.0), 4)
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


def add_future_gains(rows: list[dict], horizons: Iterable[int]) -> list[dict]:
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(str(row.get("file", "")), as_int(row, "team"))].append(row)

    for group in by_key.values():
        group.sort(key=lambda item: as_int(item, "turn"))
        by_turn = {as_int(row, "turn"): row for row in group}
        for row in group:
            turn = as_int(row, "turn")
            for horizon in horizons:
                future = by_turn.get(turn + horizon, group[-1])
                row[f"future_city_tiles_gain_{horizon}"] = max(
                    as_int(future, "city_tiles") - as_int(row, "city_tiles"), 0
                )
            future_20 = by_turn.get(turn + 20, group[-1])
            row["future_workers_gain_20"] = max(as_int(future_20, "workers") - as_int(row, "workers"), 0)
            row["future_research_gain_20"] = max(as_int(future_20, "research") - as_int(row, "research"), 0)
    return rows


def add_opponent_final_metrics(rows: list[dict]) -> list[dict]:
    finals = {}
    for row in rows:
        finals[(str(row.get("file", "")), as_int(row, "team"))] = (
            as_int(row, "final_city_tiles"),
            as_int(row, "final_units"),
        )
    for row in rows:
        own_key = (str(row.get("file", "")), as_int(row, "team"))
        opp_key = (own_key[0], 1 - own_key[1])
        own_tiles, own_units = finals.get(own_key, (as_int(row, "final_city_tiles"), as_int(row, "final_units")))
        opp_tiles, opp_units = finals.get(opp_key, (0, 0))
        row["final_opponent_city_tiles"] = opp_tiles
        row["final_opponent_units"] = opp_units
        row["final_city_tile_margin"] = own_tiles - opp_tiles
        row["final_unit_margin"] = own_units - opp_units
    return rows


def label_row(row: dict, args: argparse.Namespace) -> dict:
    turn = as_int(row, "turn")
    rank = str(row.get("rank", "")).strip()
    # Kaggle Lux rewards are final scores, not signed win/loss rewards. A losing
    # team can still have a positive reward, so only use reward as a last-resort
    # fallback when rank is unavailable.
    if rank in {"1", "1.0"}:
        win, loss = 1, 0
    elif rank in {"2", "2.0"}:
        win, loss = 0, 1
    else:
        reward = row.get("reward", "")
        win = int(str(reward).strip() not in {"", "None"} and as_float(row, "reward") > 0)
        loss = int(str(reward).strip() not in {"", "None"} and as_float(row, "reward") < 0)

    future_loss_10 = as_int(row, "future_team_loss_10")
    future_loss_20 = as_int(row, "future_team_loss_20")
    future_loss_30 = as_int(row, "future_team_loss_30")
    low_pre_night = int(as_int(row, "pre_night") and as_float(row, "p25_city_fuel_turns") < args.low_fuel_turns)
    low_night = int(as_int(row, "is_night") and as_float(row, "p25_city_fuel_turns") < args.low_fuel_turns)
    scale_without_buffer = int(
        as_float(row, "city_tiles_growth_10") > 0
        and as_float(row, "fuel_turns_total_drop_10") >= args.fuel_drop_threshold
        and as_float(row, "p25_city_fuel_turns") < args.safe_fuel_turns
    )

    expansion_taken = int(as_int(row, "bcity_actions") > 0 or as_int(row, "city_tiles_delta_next") > 0)
    expansion_safe_20 = int(
        expansion_taken
        and future_loss_20 == 0
        and as_int(row, "future_city_tiles_gain_20") >= 1
        and as_float(row, "p25_city_fuel_turns") >= args.low_fuel_turns
    )
    expansion_success_40 = int(
        expansion_taken
        and as_int(row, "future_team_loss_40") == 0
        and as_int(row, "future_city_tiles_gain_40") >= args.expansion_gain_threshold
        and (win or as_int(row, "final_city_tile_margin") >= 0)
    )
    expansion_safe_window = int(
        not expansion_taken
        and as_int(row, "unit_cap_margin") > 0
        and as_float(row, "p25_city_fuel_turns") >= args.safe_fuel_turns
        and future_loss_20 == 0
        and as_int(row, "turns_remaining") >= 40
    )

    success_survived = int(as_int(row, "final_city_tiles") > 0)
    success_scale_advantage = int(as_int(row, "final_city_tile_margin") > 0)
    success_stable_scale = int(
        success_survived
        and as_int(row, "final_city_tiles") >= args.success_min_city_tiles
        and as_int(row, "future_team_loss_30") == 0
    )

    error_failed_big = int(loss and future_loss_20 >= args.big_loss_threshold)
    error_low_fuel_bw = int(as_int(row, "bw_low_fuel_lt5_actions") > 0 and future_loss_10 > 0)
    error_low_fuel_bcity = int(as_int(row, "bcity_adjacent_low_fuel_lt5_actions") > 0 and future_loss_10 > 0)
    error_late_research = int(turn >= args.late_turn and as_int(row, "research_actions") > 0 and future_loss_20 > 0)
    error_scale_fuel_drop = int(scale_without_buffer and future_loss_20 > 0)

    if error_failed_big or error_low_fuel_bw or error_low_fuel_bcity or error_scale_fuel_drop:
        strategy_label = "error"
    elif future_loss_20 > 0 or low_pre_night or low_night or scale_without_buffer:
        strategy_label = "risk"
    elif expansion_success_40 or expansion_safe_20:
        strategy_label = "expansion_success"
    elif win and success_survived:
        strategy_label = "success"
    else:
        strategy_label = "neutral"

    sample_weight = 1.0
    if strategy_label == "error":
        sample_weight += 1.0
    elif strategy_label == "risk":
        sample_weight += 0.5
    elif strategy_label == "expansion_success":
        sample_weight += 0.35
    elif strategy_label == "success":
        sample_weight += 0.15

    row.update(
        {
            "label_version": args.label_version,
            "source_kind": source_kind(str(row.get("file", ""))),
            "source_opponent": source_opponent_from_file(str(row.get("file", ""))),
            "eval_side": eval_side_from_file(str(row.get("file", ""))),
            "night_cycle": turn // 40,
            "win_label": win,
            "loss_label": loss,
            "risk_city_loss_10": int(future_loss_10 > 0),
            "risk_city_loss_20": int(future_loss_20 > 0),
            "risk_city_loss_30": int(future_loss_30 > 0),
            "risk_big_loss_20": int(future_loss_20 >= args.big_loss_threshold),
            "risk_low_fuel_pre_night": low_pre_night,
            "risk_low_fuel_night": low_night,
            "risk_scale_without_buffer": scale_without_buffer,
            "error_failed_with_big_loss": error_failed_big,
            "error_low_fuel_bw_then_loss": error_low_fuel_bw,
            "error_low_fuel_bcity_then_loss": error_low_fuel_bcity,
            "error_late_research_then_loss": error_late_research,
            "error_scale_growth_fuel_drop_then_loss": error_scale_fuel_drop,
            "expansion_taken": expansion_taken,
            "expansion_safe_20": expansion_safe_20,
            "expansion_success_40": expansion_success_40,
            "expansion_failed_20": int(expansion_taken and future_loss_20 > 0),
            "expansion_safe_window_proxy": expansion_safe_window,
            "success_survived": success_survived,
            "success_win": win,
            "success_scale_advantage": success_scale_advantage,
            "success_stable_scale": success_stable_scale,
            "strategy_label": strategy_label,
            "sample_weight": round(sample_weight, 4),
        }
    )
    return {field: row.get(field, "") for field in FIELDNAMES}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path | None, rows: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def group_summary(data: pd.DataFrame, key: str) -> dict:
    if key not in data or data.empty:
        return {}
    return (
        data.groupby(key)
        .agg(
            rows=("strategy_label", "size"),
            risk_rate=("risk_city_loss_20", "mean"),
            error_rate=("error_failed_with_big_loss", "mean"),
            expansion_success_rate=("expansion_success_40", "mean"),
            win_rate=("success_win", "mean"),
        )
        .to_dict("index")
    )


def manifest_summary(manifest_rows: list[dict]) -> dict:
    if not manifest_rows:
        return {}
    data = pd.DataFrame(manifest_rows)
    return {
        "candidate_unique_paths": int(len(data)),
        "by_status": data["status"].value_counts().to_dict(),
        "by_reason": data["reason"].value_counts().to_dict(),
        "by_manifest_source_kind": data["source_kind"].value_counts().to_dict(),
        "by_manifest_map_size": data["map_size"].value_counts().to_dict(),
    }


def write_summary(
    path: Path,
    rows: list[dict],
    replay_count: int,
    skipped_count: int,
    manifest_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    summary = {
        "label_version": args.label_version,
        "replays": replay_count,
        "skipped_replays": skipped_count,
        "rows": len(rows),
        "map_sizes": args.map_sizes,
        "big_loss_threshold": args.big_loss_threshold,
        "low_fuel_turns": args.low_fuel_turns,
        "safe_fuel_turns": args.safe_fuel_turns,
    }
    summary["manifest"] = manifest_summary(manifest_rows)
    if rows:
        data = pd.DataFrame(rows)
        for column in LABEL_FIELDS:
            if column not in {"strategy_label"}:
                data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
        summary.update(
            {
                "strategy_label_counts": data["strategy_label"].value_counts().to_dict(),
                "risk_city_loss_20_rate": float(data["risk_city_loss_20"].mean()),
                "risk_big_loss_20_rate": float(data["risk_big_loss_20"].mean()),
                "error_failed_with_big_loss_rate": float(data["error_failed_with_big_loss"].mean()),
                "expansion_success_40_rate": float(data["expansion_success_40"].mean()),
                "success_win_rate": float(data["success_win"].mean()),
                "by_source_kind": group_summary(data, "source_kind"),
                "by_map_size": group_summary(data, "map_size"),
                "by_eval_side": group_summary(data, "eval_side"),
                "by_source_opponent": group_summary(data, "source_opponent"),
                "by_team_name": group_summary(data, "team_name"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_patterns(values: list[str]) -> list[str]:
    return values if values else DEFAULT_PATTERNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified Lux replay strategy labels.")
    parser.add_argument("patterns", nargs="*", help="Replay glob patterns. Defaults include dataset/raw and diagnostic replays.")
    parser.add_argument("--output-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1_summary.json"))
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--label-version", default="strategy_labels_v1")
    parser.add_argument("--map-sizes", default="", help="Comma-separated map sizes to keep. Empty keeps all.")
    parser.add_argument("--team-names", default="", help="Comma-separated exact team names to keep. Empty keeps both teams.")
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--big-loss-threshold", type=int, default=10)
    parser.add_argument("--low-fuel-turns", type=float, default=5.0)
    parser.add_argument("--safe-fuel-turns", type=float, default=15.0)
    parser.add_argument("--fuel-drop-threshold", type=float, default=8.0)
    parser.add_argument("--late-turn", type=int, default=160)
    parser.add_argument("--success-min-city-tiles", type=int, default=10)
    parser.add_argument("--expansion-gain-threshold", type=int, default=2)
    parser.add_argument("--trend-window", type=int, default=10)
    args = parser.parse_args()

    map_filter = parse_csv_ints(args.map_sizes)
    team_filter = parse_csv_strings(args.team_names)
    horizons = [5, 10, 20, 30, 40]
    labeled_rows: list[dict] = []
    manifest_rows: list[dict] = []
    replay_count = 0
    skipped_count = 0

    for path in iter_replay_paths(parse_patterns(args.patterns)):
        bundle = load_replay_bundle(path)
        if bundle is None:
            manifest_rows.append(
                {
                    "file": str(path),
                    "status": "skipped",
                    "reason": "invalid_or_unsupported_replay_json",
                    "source_kind": source_kind(str(path)),
                    "source_format": "",
                    "submission_id": "",
                    "episode_id": "",
                    "map_size": "",
                    "width": "",
                    "height": "",
                    "team_0_name": "",
                    "team_1_name": "",
                    "rows_written": 0,
                }
            )
            skipped_count += 1
            continue
        if map_filter and int(bundle["width"]) not in map_filter:
            manifest_rows.append(
                {
                    "file": str(path),
                    "status": "skipped",
                    "reason": "map_size_filtered",
                    "source_kind": source_kind(str(path)),
                    "source_format": bundle["source_format"],
                    "submission_id": bundle["submission_id"],
                    "episode_id": bundle["episode_id"],
                    "map_size": int(bundle["width"]),
                    "width": int(bundle["width"]),
                    "height": int(bundle["height"]),
                    "team_0_name": bundle["team_names"][0] if len(bundle["team_names"]) > 0 else "",
                    "team_1_name": bundle["team_names"][1] if len(bundle["team_names"]) > 1 else "",
                    "rows_written": 0,
                }
            )
            continue
        if team_filter and not any(name in team_filter for name in bundle["team_names"]):
            manifest_rows.append(
                {
                    "file": str(path),
                    "status": "skipped",
                    "reason": "team_name_filtered",
                    "source_kind": source_kind(str(path)),
                    "source_format": bundle["source_format"],
                    "submission_id": bundle["submission_id"],
                    "episode_id": bundle["episode_id"],
                    "map_size": int(bundle["width"]),
                    "width": int(bundle["width"]),
                    "height": int(bundle["height"]),
                    "team_0_name": bundle["team_names"][0] if len(bundle["team_names"]) > 0 else "",
                    "team_1_name": bundle["team_names"][1] if len(bundle["team_names"]) > 1 else "",
                    "rows_written": 0,
                }
            )
            continue

        rows = rows_for_bundle(bundle, horizons)
        if team_filter:
            rows = [row for row in rows if row.get("team_name", "") in team_filter]
        rows = add_trends(rows, args.trend_window)
        rows = add_future_gains(rows, [10, 20, 40])
        rows = add_opponent_final_metrics(rows)
        labeled_rows.extend(label_row(row, args) for row in rows)
        manifest_rows.append(
            {
                "file": str(path),
                "status": "included",
                "reason": "included",
                "source_kind": source_kind(str(path)),
                "source_format": bundle["source_format"],
                "submission_id": bundle["submission_id"],
                "episode_id": bundle["episode_id"],
                "map_size": int(bundle["width"]),
                "width": int(bundle["width"]),
                "height": int(bundle["height"]),
                "team_0_name": bundle["team_names"][0] if len(bundle["team_names"]) > 0 else "",
                "team_1_name": bundle["team_names"][1] if len(bundle["team_names"]) > 1 else "",
                "rows_written": len(rows),
            }
        )

        replay_count += 1
        if args.max_replays and replay_count >= args.max_replays:
            break

    write_csv(args.output_csv, labeled_rows)
    write_manifest(args.manifest_csv, manifest_rows)
    write_summary(args.summary_json, labeled_rows, replay_count, skipped_count, manifest_rows, args)
    print(f"replays: {replay_count}")
    print(f"skipped: {skipped_count}")
    print(f"rows: {len(labeled_rows)}")
    print(f"labels: {args.output_csv}")
    print(f"summary: {args.summary_json}")
    if args.manifest_csv is not None:
        print(f"manifest: {args.manifest_csv}")


if __name__ == "__main__":
    main()
