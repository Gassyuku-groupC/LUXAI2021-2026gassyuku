#!/usr/bin/env python3
"""Reweight imitation rows with soft risk labels instead of hard action bans."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_action_risk_features import (  # noqa: E402
    DAY_LEN,
    DN_CYCLE_LEN,
    adjacent_cities,
    city_at,
    near_resource,
    state_from_updates,
    team_units,
)


def parse_int_set(text: str) -> set[int]:
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def fuel_turns(city: dict) -> float:
    return float(city.get("fuel", 0.0)) / max(float(city.get("lightupkeep", 1.0)), 1.0)


def is_risk_window(turn: int, lead_turns: int) -> bool:
    return turn % DN_CYCLE_LEN >= DAY_LEN - lead_turns


def is_pre_night_window(turn: int, lead_turns: int) -> bool:
    cycle_turn = turn % DN_CYCLE_LEN
    return DAY_LEN - lead_turns <= cycle_turn < DAY_LEN


def state_for_step(replay: dict, row: dict, step: int, state_cache: dict[int, dict]) -> dict:
    if step not in state_cache:
        obs = replay["steps"][step][0]["observation"]
        state_cache[step] = state_from_updates(
            list(obs.get("updates") or []),
            int(obs.get("width") or row["width"]),
            int(obs.get("height") or row["height"]),
        )
    return state_cache[step]


def team_city_tiles(state: dict, team: int) -> int:
    return sum(
        len(city.get("cityCells") or [])
        for city in (state.get("cities") or {}).values()
        if int(city.get("team", -1)) == team
    )


def future_team_loss(replay: dict, row: dict, state_cache: dict[int, dict], horizon: int) -> int:
    state_step = int(row["state_step"])
    teacher_player = int(row["teacher_player"])
    max_dt = min(horizon, len(replay.get("steps") or []) - state_step - 1)
    if max_dt <= 0:
        return 0
    now = team_city_tiles(state_for_step(replay, row, state_step, state_cache), teacher_player)
    worst_loss = 0
    for dt in range(1, max_dt + 1):
        future = team_city_tiles(state_for_step(replay, row, state_step + dt, state_cache), teacher_player)
        worst_loss = max(worst_loss, now - future)
    return worst_loss


def min_city_fuel_turns(state: dict, team: int) -> float | None:
    values = [
        fuel_turns(city)
        for city in (state.get("cities") or {}).values()
        if int(city.get("team", -1)) == team
    ]
    return min(values) if values else None


def severity_scale(severity: int, args: argparse.Namespace) -> tuple[float, str]:
    if severity >= 3:
        return args.severe_scale, f"risk_severex{args.severe_scale:g}"
    if severity == 2:
        return args.high_scale, f"risk_highx{args.high_scale:g}"
    if severity == 1:
        return args.moderate_scale, f"risk_moderatex{args.moderate_scale:g}"
    return 1.0, "risk_none"


def row_risk(
    *,
    replay: dict,
    row: dict,
    state_cache: dict[int, dict],
    args: argparse.Namespace,
) -> tuple[int, list[str]]:
    state_step = int(row["state_step"])
    action_step = int(row["action_step"])
    teacher_player = int(row["teacher_player"])
    if not is_risk_window(state_step, args.risk_window_lead_turns):
        return 0, ["outside_risk_window"]

    state = state_for_step(replay, row, state_step, state_cache)
    actions = replay["steps"][action_step][teacher_player].get("action") or []
    severity = 0
    reasons = []

    for action in actions:
        parts = str(action).split()
        if not parts:
            continue
        if parts[0] == "bw" and len(parts) >= 3:
            try:
                x, y = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            target = city_at(state, teacher_player, x, y)
            if target is None:
                continue
            value = fuel_turns(target[1])
            if value < args.bw_severe_fuel_turns:
                severity = max(severity, 3)
                reasons.append(f"bw_fuel<{args.bw_severe_fuel_turns:g}")
                if is_pre_night_window(state_step, args.risk_window_lead_turns):
                    reasons.append("pre_night_bw_fuel_severe")
            elif value < args.bw_high_fuel_turns:
                severity = max(severity, 2)
                reasons.append(f"bw_fuel<{args.bw_high_fuel_turns:g}")
            elif value < args.bw_moderate_fuel_turns:
                severity = max(severity, 1)
                reasons.append(f"bw_fuel<{args.bw_moderate_fuel_turns:g}")
        elif parts[0] == "bcity" and len(parts) >= 2:
            unit = team_units(state, teacher_player).get(parts[1])
            if not unit:
                continue
            pos = (int(unit["x"]), int(unit["y"]))
            adjacent = adjacent_cities(state, teacher_player, pos)
            if adjacent:
                value = min(fuel_turns(city) for _, city in adjacent)
                if value < args.bcity_severe_adjacent_fuel_turns:
                    severity = max(severity, 3)
                    reasons.append(f"bcity_adj_fuel<{args.bcity_severe_adjacent_fuel_turns:g}")
                elif value < args.bcity_high_adjacent_fuel_turns:
                    severity = max(severity, 2)
                    reasons.append(f"bcity_adj_fuel<{args.bcity_high_adjacent_fuel_turns:g}")
            elif near_resource(state, pos):
                reasons.append("bcity_isolated_resource")
            else:
                reasons.append("bcity_isolated_nonresource")

    return severity, reasons or ["no_risky_action"]


def reweight_row(row: dict, replay: dict, state_cache: dict[int, dict], args: argparse.Namespace) -> dict:
    original_weight = float(row["weight"])
    weight = original_weight
    reasons = [row.get("weight_reason", "")]
    severity, risk_reasons = row_risk(replay=replay, row=row, state_cache=state_cache, args=args)
    scale, scale_reason = severity_scale(severity, args)
    weight *= scale
    reasons.append(scale_reason)
    reasons.extend(risk_reasons)
    state_step = int(row["state_step"])
    teacher_player = int(row["teacher_player"])
    future_loss = future_team_loss(replay, row, state_cache, args.future_loss_horizon)
    state = state_for_step(replay, row, state_step, state_cache)
    min_fuel = min_city_fuel_turns(state, teacher_player)

    if "pre_night_bw_fuel_severe" in risk_reasons:
        weight *= args.pre_night_bw_severe_extra_scale
        reasons.append(f"pre_night_bw_severex{args.pre_night_bw_severe_extra_scale:g}")

    if int(row.get("night_city_loss_next", 0)) > 0:
        weight *= args.next_loss_scale
        reasons.append(f"next_lossx{args.next_loss_scale:g}")
    elif is_risk_window(int(row["state_step"]), args.risk_window_lead_turns) and severity == 0:
        weight *= args.safe_risk_window_scale
        reasons.append(f"safe_risk_windowx{args.safe_risk_window_scale:g}")

    if (
        severity == 0
        and future_loss == 0
        and min_fuel is not None
        and is_pre_night_window(state_step, args.risk_window_lead_turns)
        and min_fuel >= args.safe_pre_night_min_fuel_turns
    ):
        weight *= args.safe_pre_night_buffer_scale
        reasons.append(f"safe_pre_night_bufferx{args.safe_pre_night_buffer_scale:g}")

    if severity == 0 and "bcity_isolated_resource" in risk_reasons and int(row.get("night_city_loss_next", 0)) == 0:
        weight *= args.resource_backup_city_scale
        reasons.append(f"resource_backup_cityx{args.resource_backup_city_scale:g}")

    row = dict(row)
    row["original_weight"] = f"{original_weight:.4f}"
    row["risk_severity"] = str(severity)
    row[f"future_team_loss_{args.future_loss_horizon}"] = str(future_loss)
    row["min_city_fuel_turns"] = "" if min_fuel is None else f"{min_fuel:.4f}"
    row["risk_scale"] = f"{(weight / original_weight if original_weight else 0.0):.4f}"
    row["weight"] = f"{min(max(weight, args.min_weight), args.max_weight):.4f}"
    row["weight_reason"] = "+".join(part for part in reasons if part)
    return row


def grouped_rows(path: Path, map_sizes: set[int], max_rows: int) -> dict[Path, list[dict]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            if map_sizes and int(row["width"]) not in map_sizes:
                continue
            grouped[Path(row["file"])].append(row)
            if max_rows and sum(len(items) for items in grouped.values()) >= max_rows:
                break
    return dict(grouped)


def write_rows(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["original_weight", "risk_severity", "future_team_loss_10", "min_city_fuel_turns", "risk_scale"]
    out_fields = [field for field in fieldnames if field not in extra] + extra
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a soft risk-aware imitation index.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_hq.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_risk_aware_v1.csv"))
    parser.add_argument("--map-sizes", default="", help="Comma-separated map sizes to keep; empty keeps all.")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--risk-window-lead-turns", type=int, default=5)
    parser.add_argument("--bw-severe-fuel-turns", type=float, default=3.0)
    parser.add_argument("--bw-high-fuel-turns", type=float, default=5.0)
    parser.add_argument("--bw-moderate-fuel-turns", type=float, default=10.0)
    parser.add_argument("--bcity-severe-adjacent-fuel-turns", type=float, default=3.0)
    parser.add_argument("--bcity-high-adjacent-fuel-turns", type=float, default=5.0)
    parser.add_argument("--severe-scale", type=float, default=0.20)
    parser.add_argument("--high-scale", type=float, default=0.40)
    parser.add_argument("--moderate-scale", type=float, default=0.70)
    parser.add_argument("--next-loss-scale", type=float, default=0.50)
    parser.add_argument("--safe-risk-window-scale", type=float, default=1.20)
    parser.add_argument("--future-loss-horizon", type=int, default=10)
    parser.add_argument("--safe-pre-night-min-fuel-turns", type=float, default=10.0)
    parser.add_argument("--safe-pre-night-buffer-scale", type=float, default=1.0)
    parser.add_argument("--pre-night-bw-severe-extra-scale", type=float, default=1.0)
    parser.add_argument("--resource-backup-city-scale", type=float, default=1.10)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=4.0)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as in_file:
        fieldnames = csv.DictReader(in_file).fieldnames or []
    grouped = grouped_rows(args.input, parse_int_set(args.map_sizes), args.max_rows)

    rows = []
    for replay_path, replay_rows in grouped.items():
        with replay_path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
        state_cache: dict[int, dict] = {}
        for row in replay_rows:
            rows.append(reweight_row(row, replay, state_cache, args))

    write_rows(args.output, rows, fieldnames)
    total_weight = sum(float(row["weight"]) for row in rows)
    severity_counts = defaultdict(int)
    for row in rows:
        severity_counts[int(row["risk_severity"])] += 1
    print(f"rows: {len(rows)}")
    print(f"mean weight: {total_weight / len(rows):.3f}" if rows else "mean weight: n/a")
    print("risk severity:", dict(sorted(severity_counts.items())))
    print(f"index: {args.output}")


if __name__ == "__main__":
    main()
