#!/usr/bin/env python3
"""Extract city-level risk features from stateful Lux replays."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DAY_LEN = 30
NIGHT_LEN = 10
DN_CYCLE_LEN = DAY_LEN + NIGHT_LEN


def iter_replay_paths(patterns: List[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for raw_path in glob.glob(pattern, recursive=True):
            path = Path(raw_path)
            if path.name.endswith(".commands.json"):
                continue
            if path.name.endswith(".log"):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def load_stateful_replay(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
    except Exception:
        return None
    if isinstance(replay, dict) and replay.get("stateful") and isinstance(replay.get("stateful"), list):
        return replay
    return None


def city_tile_counts(state: dict, team: int) -> Dict[str, int]:
    counts = {}
    for city_id, city in state.get("cities", {}).items():
        if int(city.get("team", -1)) == team:
            counts[city_id] = len(city.get("cityCells") or [])
    return counts


def team_city_tiles(state: dict, team: int) -> int:
    return sum(city_tile_counts(state, team).values())


def team_units(state: dict, team: int) -> Dict[str, dict]:
    return state.get("teamStates", {}).get(str(team), {}).get("units", {}) or {}


def unit_fuel(unit: dict) -> int:
    cargo = unit.get("cargo", {}) or {}
    return int(cargo.get("wood", 0)) + int(cargo.get("coal", 0)) * 10 + int(cargo.get("uranium", 0)) * 40


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def city_positions(city: dict) -> set[Tuple[int, int]]:
    return {(int(cell["x"]), int(cell["y"])) for cell in city.get("cityCells") or []}


def near_resource(state: dict, pos: Tuple[int, int], radius: int = 2) -> bool:
    game_map = state.get("map") or []
    height = len(game_map)
    width = len(game_map[0]) if height else 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            x, y = pos[0] + dx, pos[1] + dy
            if not (0 <= x < width and 0 <= y < height):
                continue
            resource = (game_map[y][x] or {}).get("resource")
            if resource and int(resource.get("amount", 0)) > 0:
                return True
    return False


def parse_build_city_actions(replay: dict, turn: int, team: int) -> List[str]:
    commands = replay.get("allCommands") or []
    if turn >= len(commands):
        return []
    return [
        command.get("command", "")
        for command in commands[turn]
        if int(command.get("agentID", -1)) == team and str(command.get("command", "")).startswith("bcity ")
    ]


def build_city_contexts(state: dict, replay: dict, turn: int, team: int) -> List[dict]:
    cities = {
        city_id: city
        for city_id, city in (state.get("cities") or {}).items()
        if int(city.get("team", -1)) == team
    }
    units = team_units(state, team)
    contexts = []
    for action in parse_build_city_actions(replay, turn, team):
        parts = action.split()
        if len(parts) < 2:
            continue
        unit = units.get(parts[1])
        if not unit:
            continue
        pos = (int(unit["x"]), int(unit["y"]))
        adjacent_city_ids = [
            city_id
            for city_id, city in cities.items()
            if any(manhattan(pos, city_pos) == 1 for city_pos in city_positions(city))
        ]
        contexts.append(
            {
                "unit_id": parts[1],
                "pos": pos,
                "unit_fuel": unit_fuel(unit),
                "near_resource": near_resource(state, pos),
                "isolated": not adjacent_city_ids,
                "adjacent_city_ids": adjacent_city_ids,
            }
        )
    return contexts


def future_team_loss(states: List[dict], turn: int, team: int, horizon: int) -> int:
    if turn + horizon >= len(states):
        horizon = len(states) - turn - 1
    if horizon <= 0:
        return 0
    now = team_city_tiles(states[turn], team)
    worst_loss = 0
    for dt in range(1, horizon + 1):
        future = team_city_tiles(states[turn + dt], team)
        worst_loss = max(worst_loss, now - future)
    return worst_loss


def city_future_loss(states: List[dict], turn: int, team: int, city_id: str, horizon: int) -> int:
    now = city_tile_counts(states[turn], team).get(city_id, 0)
    worst_loss = 0
    max_dt = min(horizon, len(states) - turn - 1)
    for dt in range(1, max_dt + 1):
        future = city_tile_counts(states[turn + dt], team).get(city_id, 0)
        worst_loss = max(worst_loss, now - future)
    return worst_loss


def rows_for_replay(path: Path, replay: dict, horizons: List[int]) -> List[dict]:
    rows = []
    states = replay["stateful"]
    for turn, state in enumerate(states[:-1]):
        cycle_turn = turn % DN_CYCLE_LEN
        is_night = int(cycle_turn >= DAY_LEN)
        pre_night = int(DAY_LEN - 5 <= cycle_turn < DAY_LEN)
        for team in (0, 1):
            contexts = build_city_contexts(state, replay, turn, team)
            isolated_builds = sum(int(ctx["isolated"]) for ctx in contexts)
            resource_builds = sum(int(ctx["near_resource"]) for ctx in contexts)
            adjacent_by_city = {}
            for ctx in contexts:
                for city_id in ctx["adjacent_city_ids"]:
                    adjacent_by_city[city_id] = adjacent_by_city.get(city_id, 0) + 1

            team_counts_now = team_city_tiles(state, team)
            team_counts_next = team_city_tiles(states[turn + 1], team)
            upkeep_increased = int(team_counts_next > team_counts_now)
            for city_id, city in (state.get("cities") or {}).items():
                if int(city.get("team", -1)) != team:
                    continue
                size = len(city.get("cityCells") or [])
                fuel = float(city.get("fuel", 0))
                upkeep = float(city.get("lightupkeep", 0))
                fuel_turns = fuel / max(upkeep, 1.0)
                row = {
                    "file": str(path),
                    "turn": turn,
                    "team": team,
                    "city_id": city_id,
                    "map_size": replay.get("width", ""),
                    "cycle_turn": cycle_turn,
                    "is_night": is_night,
                    "pre_night": pre_night,
                    "city_size": size,
                    "city_fuel": round(fuel, 3),
                    "city_upkeep": round(upkeep, 3),
                    "fuel_turns": round(fuel_turns, 3),
                    "team_city_tiles": team_counts_now,
                    "team_units": len(team_units(state, team)),
                    "build_city_actions": len(contexts),
                    "isolated_build_city_actions": isolated_builds,
                    "resource_near_build_city_actions": resource_builds,
                    "adjacent_build_city_actions": adjacent_by_city.get(city_id, 0),
                    "upkeep_increased_next": upkeep_increased,
                }
                for horizon in horizons:
                    row[f"future_team_loss_{horizon}"] = future_team_loss(states, turn, team, horizon)
                    row[f"future_city_loss_{horizon}"] = city_future_loss(states, turn, team, city_id, horizon)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract city risk features from stateful replays.")
    parser.add_argument("patterns", nargs="+", help="Replay glob patterns.")
    parser.add_argument("--output", type=Path, default=Path("outputs/risk_feature_logs/city_risk_features.csv"))
    parser.add_argument("--horizons", default="1,3,5,10")
    args = parser.parse_args()

    horizons = [int(part.strip()) for part in args.horizons.split(",") if part.strip()]
    all_rows = []
    replay_count = 0
    for path in iter_replay_paths(args.patterns):
        replay = load_stateful_replay(path)
        if replay is None:
            continue
        replay_count += 1
        all_rows.extend(rows_for_replay(path, replay, horizons))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        args.output.write_text("", encoding="utf-8")
        print("replays: 0")
        print("rows: 0")
        print(f"output: {args.output}")
        return

    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    risky = sum(int(row["future_team_loss_10"]) > 0 for row in all_rows)
    print(f"replays: {replay_count}")
    print(f"rows: {len(all_rows)}")
    print(f"rows with future_team_loss_10 > 0: {risky}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
