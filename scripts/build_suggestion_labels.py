#!/usr/bin/env python3
"""Build counterfactual suggestion labels from Lux replay data.

The first label type is deliberately non-invasive:

* suggest fuel support when a cargo-carrying worker is adjacent to a low-fuel
  friendly city near night/night.
* penalize only if the suggestion was ignored and the team lost city tiles in
  the next N turns.
* lightly reward accepted suggestions only when the team did not lose city
  tiles in the next N turns.
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

from extract_strategy_features import iter_replay_paths, load_replay_bundle  # noqa: E402
from mine_adjacent_fuel_support import mine_bundle  # noqa: E402


DEFAULT_PATTERNS = [
    "outputs/diagnostic_layer/best_agent_public_opponents_v1_16/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_public_opponents_v2_16/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_opponent_pool_v1/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_v10_random2_16_allopponents/replays/map_16x16_vs_*.json",
    "outputs/diagnostic_layer/best_agent_seed1259068876_16_baseline/replays/map_16x16_vs_*.json",
]


FIELDNAMES = [
    "suggest_type",
    "label_version",
    "source_file",
    "episode_id",
    "map_size",
    "source_opponent",
    "opponent_name",
    "team",
    "eval_side",
    "turn",
    "cycle_turn",
    "turns_to_night",
    "is_night",
    "unit_id",
    "unit_x",
    "unit_y",
    "unit_cargo_fuel",
    "action_taken",
    "suggested_action",
    "suggested_city_id",
    "suggested_city_x",
    "suggested_city_y",
    "city_fuel_turns",
    "city_fuel",
    "city_upkeep",
    "city_tiles",
    "team_city_tiles",
    "final_city_tiles",
    "future_team_loss_5",
    "future_team_loss_10",
    "future_team_loss_20",
    "ignored_suggestion",
    "penalty_label",
    "penalty_weight",
    "positive_label",
    "positive_weight",
    "reward_value",
    "outcome_label",
]


def as_float(row: dict, key: str) -> float:
    return float(row.get(key, 0) or 0)


def as_int(row: dict, key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def eval_side_from_file(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_p0"):
        return "0"
    if stem.endswith("_p1"):
        return "1"
    return ""


def source_opponent_from_file(path: str) -> str:
    stem = Path(path).stem
    match = re.match(r"map_\d+x\d+_vs_(?P<opponent>.+)_\d+_p[01]$", stem)
    return match.group("opponent") if match else ""


def direction_to_city(row: dict) -> str:
    ux, uy = as_int(row, "unit_x"), as_int(row, "unit_y")
    cx, cy = as_int(row, "city_x"), as_int(row, "city_y")
    dx, dy = cx - ux, cy - uy
    if dx == 1 and dy == 0:
        return "e"
    if dx == -1 and dy == 0:
        return "w"
    if dx == 0 and dy == 1:
        return "s"
    if dx == 0 and dy == -1:
        return "n"
    return ""


def penalty_weight(loss_10: float, city_fuel_turns: float, cargo_fuel: float) -> float:
    if loss_10 <= 0:
        return 0.0
    weight = 1.0
    if loss_10 >= 5:
        weight += 0.5
    if loss_10 >= 10:
        weight += 0.5
    if city_fuel_turns < 1.5:
        weight += 0.5
    elif city_fuel_turns < 2.5:
        weight += 0.25
    if cargo_fuel >= 80:
        weight += 0.25
    if cargo_fuel >= 100:
        weight += 0.25
    return round(weight, 4)


def positive_weight(city_fuel_turns: float, cargo_fuel: float) -> float:
    weight = 0.10
    if city_fuel_turns < 1.5:
        weight += 0.05
    if cargo_fuel >= 100:
        weight += 0.05
    return round(weight, 4)


def make_label(row: dict, args: argparse.Namespace) -> dict | None:
    event_type = row.get("event_type")
    if event_type not in {"missed_adjacent_fuel", "supporting_adjacent_fuel"}:
        return None
    city_fuel_turns = as_float(row, "city_fuel_turns")
    cargo_fuel = as_float(row, "unit_cargo_fuel")
    turn = as_int(row, "turn")
    if city_fuel_turns >= args.city_fuel_turns_lt:
        return None
    if cargo_fuel < args.min_cargo_fuel:
        return None
    if turn > args.max_turn:
        return None

    direction = direction_to_city(row)
    if not direction:
        return None

    loss_10 = as_float(row, "future_team_loss_10")
    ignored = int(event_type == "missed_adjacent_fuel")
    penalty = int(ignored and loss_10 > 0)
    positive = int(not ignored and loss_10 == 0)
    p_weight = penalty_weight(loss_10, city_fuel_turns, cargo_fuel) if penalty else 0.0
    pos_weight = positive_weight(city_fuel_turns, cargo_fuel) if positive else 0.0
    if penalty:
        outcome = "ignored_then_loss"
    elif ignored:
        outcome = "ignored_without_loss"
    elif positive:
        outcome = "accepted_without_loss"
    else:
        outcome = "accepted_but_loss"
    return {
        "suggest_type": "suggest_fuel_support",
        "label_version": args.label_version,
        "source_file": row.get("file", ""),
        "episode_id": row.get("episode_id", ""),
        "map_size": row.get("map_size", ""),
        "source_opponent": source_opponent_from_file(row.get("file", "")),
        "opponent_name": row.get("opponent_name", ""),
        "team": row.get("team", ""),
        "eval_side": eval_side_from_file(row.get("file", "")),
        "turn": row.get("turn", ""),
        "cycle_turn": row.get("cycle_turn", ""),
        "turns_to_night": row.get("turns_to_night", ""),
        "is_night": row.get("is_night", ""),
        "unit_id": row.get("unit_id", ""),
        "unit_x": row.get("unit_x", ""),
        "unit_y": row.get("unit_y", ""),
        "unit_cargo_fuel": row.get("unit_cargo_fuel", ""),
        "action_taken": row.get("action", ""),
        "suggested_action": f"m {row.get('unit_id', '')} {direction}",
        "suggested_city_id": row.get("city_id", ""),
        "suggested_city_x": row.get("city_x", ""),
        "suggested_city_y": row.get("city_y", ""),
        "city_fuel_turns": row.get("city_fuel_turns", ""),
        "city_fuel": row.get("city_fuel", ""),
        "city_upkeep": row.get("city_upkeep", ""),
        "city_tiles": row.get("city_tiles", ""),
        "team_city_tiles": row.get("team_city_tiles", ""),
        "final_city_tiles": row.get("final_city_tiles", ""),
        "future_team_loss_5": row.get("future_team_loss_5", ""),
        "future_team_loss_10": row.get("future_team_loss_10", ""),
        "future_team_loss_20": row.get("future_team_loss_20", ""),
        "ignored_suggestion": ignored,
        "penalty_label": penalty,
        "penalty_weight": p_weight,
        "positive_label": positive,
        "positive_weight": pos_weight,
        "reward_value": round(pos_weight - p_weight, 4),
        "outcome_label": outcome,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, labels: list[dict], replay_count: int, skipped_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "replays": replay_count,
        "skipped_replays": skipped_count,
        "rows": len(labels),
    }
    if labels:
        data = pd.DataFrame(labels)
        data["penalty_label"] = pd.to_numeric(data["penalty_label"], errors="coerce").fillna(0)
        data["future_team_loss_10"] = pd.to_numeric(data["future_team_loss_10"], errors="coerce").fillna(0)
        data["penalty_weight"] = pd.to_numeric(data["penalty_weight"], errors="coerce").fillna(0)
        data["positive_label"] = pd.to_numeric(data["positive_label"], errors="coerce").fillna(0)
        data["positive_weight"] = pd.to_numeric(data["positive_weight"], errors="coerce").fillna(0)
        data["reward_value"] = pd.to_numeric(data["reward_value"], errors="coerce").fillna(0)
        summary |= {
            "penalty_rows": int(data["penalty_label"].sum()),
            "penalty_rate": float(data["penalty_label"].mean()),
            "positive_rows": int(data["positive_label"].sum()),
            "positive_rate": float(data["positive_label"].mean()),
            "mean_future_team_loss_10": float(data["future_team_loss_10"].mean()),
            "mean_penalty_weight": float(data["penalty_weight"].mean()),
            "mean_positive_weight": float(data["positive_weight"].mean()),
            "mean_reward_value": float(data["reward_value"].mean()),
            "by_turn_bucket": bucket_summary(data, "turn"),
            "by_eval_side": group_summary(data, "eval_side"),
            "by_source_opponent": group_summary(data, "source_opponent"),
            "by_opponent": group_summary(data, "opponent_name"),
            "by_outcome_label": group_summary(data, "outcome_label"),
        }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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


def bucket_summary(data: pd.DataFrame, turn_key: str) -> list[dict]:
    tmp = data.copy()
    tmp["turn_bucket"] = tmp[turn_key].astype(float).astype(int).map(turn_bucket)
    return group_summary(tmp, "turn_bucket")


def group_summary(data: pd.DataFrame, key: str) -> list[dict]:
    return (
        data.groupby(key, dropna=False)
        .agg(
            rows=("suggest_type", "size"),
            penalty_rate=("penalty_label", "mean"),
            positive_rate=("positive_label", "mean"),
            mean_future_team_loss_10=("future_team_loss_10", "mean"),
            mean_penalty_weight=("penalty_weight", "mean"),
            mean_positive_weight=("positive_weight", "mean"),
            mean_reward_value=("reward_value", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build non-invasive suggestion labels from Lux replays.")
    parser.add_argument("patterns", nargs="*", help="Replay glob patterns. Defaults to current best-agent diagnostic pools.")
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostic_layer/suggestion_labels_v1/suggestion_labels.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/diagnostic_layer/suggestion_labels_v1/suggestion_labels_summary.json"))
    parser.add_argument("--city-fuel-turns-lt", type=float, default=2.5)
    parser.add_argument("--min-cargo-fuel", type=float, default=80.0)
    parser.add_argument("--turns-to-night-lte", type=int, default=3)
    parser.add_argument("--max-turn", type=int, default=240)
    parser.add_argument("--include-night", action="store_true", default=True)
    parser.add_argument("--exclude-night", dest="include_night", action="store_false")
    parser.add_argument("--include-supporting", action="store_true", default=True)
    parser.add_argument("--exclude-supporting", dest="include_supporting", action="store_false")
    parser.add_argument("--all-teams", action="store_true")
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--label-version", default="fuel_support_v1")
    args = parser.parse_args()

    patterns = args.patterns or DEFAULT_PATTERNS
    labels: list[dict] = []
    replay_count = 0
    skipped_count = 0
    for replay_path in iter_replay_paths(patterns):
        if args.max_replays and replay_count >= args.max_replays:
            break
        bundle = load_replay_bundle(replay_path)
        if not bundle:
            skipped_count += 1
            continue
        replay_count += 1
        for event in mine_bundle(bundle, args):
            label = make_label(event, args)
            if label:
                labels.append(label)

    labels.sort(key=lambda row: (row["source_file"], int(row["team"]), int(row["turn"]), row["unit_id"]))
    write_csv(args.output, labels)
    write_summary(args.summary, labels, replay_count, skipped_count)
    print(json.dumps({
        "replays": replay_count,
        "skipped_replays": skipped_count,
        "rows": len(labels),
        "output": str(args.output),
        "summary": str(args.summary),
    }, indent=2))


if __name__ == "__main__":
    main()
