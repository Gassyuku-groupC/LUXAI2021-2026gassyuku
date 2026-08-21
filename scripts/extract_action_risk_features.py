#!/usr/bin/env python3
"""Extract action-level risk features for upkeep-increasing actions."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple


DAY_LEN = 30
DN_CYCLE_LEN = 40


def iter_replay_paths(patterns: list[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for raw_path in glob.glob(pattern, recursive=True):
            path = Path(raw_path)
            if path.name.endswith(".commands.json") or path.name.endswith(".log"):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def load_replay(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
    except Exception:
        return None
    if isinstance(replay, dict) and replay.get("stateful") and isinstance(replay.get("stateful"), list):
        return replay
    if isinstance(replay, dict) and isinstance(replay.get("steps"), list):
        return convert_kaggle_replay(replay)
    return None


def convert_kaggle_replay(replay: dict) -> Optional[dict]:
    steps = replay.get("steps") or []
    states = []
    commands = []
    width = 0
    height = 0
    for turn_steps in steps:
        if not turn_steps:
            continue
        obs = (turn_steps[0] or {}).get("observation") or {}
        updates = obs.get("updates") or []
        if not width:
            width = int(obs.get("width") or 0)
            height = int(obs.get("height") or 0)
        state = state_from_updates(updates, width, height)
        states.append(state)

        turn_commands = []
        for agent_id, agent_step in enumerate(turn_steps[:2]):
            for action in agent_step.get("action") or []:
                turn_commands.append({"agentID": agent_id, "command": action})
        commands.append(turn_commands)
    if not states:
        return None
    return {
        "stateful": states,
        "allCommands": commands,
        "width": width,
        "height": height,
    }


def state_from_updates(updates: list[str], width: int, height: int) -> dict:
    state = {
        "map": [[{} for _ in range(width)] for _ in range(height)],
        "cities": {},
        "teamStates": {"0": {"units": {}}, "1": {"units": {}}},
    }
    for update in updates:
        parts = str(update).split()
        if not parts:
            continue
        if parts[0] == "r" and len(parts) >= 5:
            resource_type, x, y, amount = parts[1], int(parts[2]), int(parts[3]), int(float(parts[4]))
            if 0 <= y < height and 0 <= x < width:
                state["map"][y][x]["resource"] = {"type": resource_type, "amount": amount}
        elif parts[0] == "u" and len(parts) >= 10:
            unit_type, team, unit_id = int(parts[1]), int(parts[2]), parts[3]
            x, y = int(parts[4]), int(parts[5])
            cooldown = float(parts[6])
            wood, coal, uranium = int(float(parts[7])), int(float(parts[8])), int(float(parts[9]))
            state["teamStates"].setdefault(str(team), {"units": {}})["units"][unit_id] = {
                "type": unit_type,
                "team": team,
                "id": unit_id,
                "x": x,
                "y": y,
                "cooldown": cooldown,
                "cargo": {"wood": wood, "coal": coal, "uranium": uranium},
            }
        elif parts[0] == "c" and len(parts) >= 5:
            team, city_id = int(parts[1]), parts[2]
            state["cities"][city_id] = {
                "team": team,
                "cityid": city_id,
                "fuel": float(parts[3]),
                "lightupkeep": float(parts[4]),
                "cityCells": [],
            }
        elif parts[0] == "ct" and len(parts) >= 6:
            team, city_id = int(parts[1]), parts[2]
            city = state["cities"].setdefault(
                city_id,
                {
                    "team": team,
                    "cityid": city_id,
                    "fuel": 0.0,
                    "lightupkeep": 1.0,
                    "cityCells": [],
                },
            )
            city["cityCells"].append(
                {
                    "x": int(parts[3]),
                    "y": int(parts[4]),
                    "cooldown": float(parts[5]),
                }
            )
    return state


def team_city_tiles(state: dict, team: int) -> int:
    return sum(
        len(city.get("cityCells") or [])
        for city in (state.get("cities") or {}).values()
        if int(city.get("team", -1)) == team
    )


def team_units(state: dict, team: int) -> dict:
    return state.get("teamStates", {}).get(str(team), {}).get("units", {}) or {}


def city_positions(city: dict) -> set[Tuple[int, int]]:
    return {(int(cell["x"]), int(cell["y"])) for cell in city.get("cityCells") or []}


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def unit_fuel(unit: dict) -> int:
    cargo = unit.get("cargo", {}) or {}
    return int(cargo.get("wood", 0)) + int(cargo.get("coal", 0)) * 10 + int(cargo.get("uranium", 0)) * 40


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


def city_at(state: dict, team: int, x: int, y: int) -> Optional[tuple[str, dict]]:
    for city_id, city in (state.get("cities") or {}).items():
        if int(city.get("team", -1)) != team:
            continue
        for cell in city.get("cityCells") or []:
            if int(cell["x"]) == x and int(cell["y"]) == y:
                return city_id, city
    return None


def adjacent_cities(state: dict, team: int, pos: Tuple[int, int]) -> list[tuple[str, dict]]:
    found = []
    for city_id, city in (state.get("cities") or {}).items():
        if int(city.get("team", -1)) != team:
            continue
        if any(manhattan(pos, city_pos) == 1 for city_pos in city_positions(city)):
            found.append((city_id, city))
    return found


def future_team_loss(states: list[dict], turn: int, team: int, horizon: int) -> int:
    max_dt = min(horizon, len(states) - turn - 1)
    if max_dt <= 0:
        return 0
    now = team_city_tiles(states[turn], team)
    worst_loss = 0
    for dt in range(1, max_dt + 1):
        worst_loss = max(worst_loss, now - team_city_tiles(states[turn + dt], team))
    return worst_loss


def base_row(path: Path, replay: dict, states: list[dict], turn: int, team: int, horizon: int) -> dict:
    state = states[turn]
    cycle_turn = turn % DN_CYCLE_LEN
    return {
        "file": str(path),
        "turn": turn,
        "team": team,
        "map_size": replay.get("width", ""),
        "cycle_turn": cycle_turn,
        "pre_night": int(DAY_LEN - 5 <= cycle_turn < DAY_LEN),
        "is_night": int(cycle_turn >= DAY_LEN),
        "team_city_tiles": team_city_tiles(state, team),
        "team_units": len(team_units(state, team)),
        f"future_team_loss_{horizon}": future_team_loss(states, turn, team, horizon),
    }


def rows_for_replay(path: Path, replay: dict, horizon: int) -> list[dict]:
    rows = []
    states = replay["stateful"]
    commands = replay.get("allCommands") or []
    for turn, state in enumerate(states[:-1]):
        if turn >= len(commands):
            continue
        for command in commands[turn]:
            team = int(command.get("agentID", -1))
            if team not in (0, 1):
                continue
            raw = str(command.get("command", ""))
            parts = raw.split()
            if not parts or parts[0] not in ("bcity", "bw", "bc"):
                continue

            row = base_row(path, replay, states, turn, team, horizon)
            row["action"] = parts[0]
            row["raw_action"] = raw
            row.update({
                "unit_fuel": "",
                "near_resource": "",
                "isolated": "",
                "adjacent_city_count": "",
                "target_city_id": "",
                "target_city_size": "",
                "target_city_fuel_turns": "",
                "min_adjacent_city_fuel_turns": "",
            })

            if parts[0] == "bcity" and len(parts) >= 2:
                unit = team_units(state, team).get(parts[1])
                if unit:
                    pos = (int(unit["x"]), int(unit["y"]))
                    adjacent = adjacent_cities(state, team, pos)
                    row["unit_fuel"] = unit_fuel(unit)
                    row["near_resource"] = int(near_resource(state, pos))
                    row["isolated"] = int(not adjacent)
                    row["adjacent_city_count"] = len(adjacent)
                    if adjacent:
                        turns = [city["fuel"] / max(city["lightupkeep"], 1) for _, city in adjacent]
                        row["min_adjacent_city_fuel_turns"] = round(min(turns), 3)
                        if len(adjacent) == 1:
                            city_id, city = adjacent[0]
                            row["target_city_id"] = city_id
                            row["target_city_size"] = len(city.get("cityCells") or [])
                            row["target_city_fuel_turns"] = round(turns[0], 3)
            elif parts[0] in ("bw", "bc") and len(parts) >= 3:
                try:
                    x, y = int(parts[1]), int(parts[2])
                except ValueError:
                    x, y = -1, -1
                target = city_at(state, team, x, y)
                if target:
                    city_id, city = target
                    row["target_city_id"] = city_id
                    row["target_city_size"] = len(city.get("cityCells") or [])
                    row["target_city_fuel_turns"] = round(city["fuel"] / max(city["lightupkeep"], 1), 3)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract action-level risk rows.")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("outputs/risk_feature_logs/action_risk_features.csv"))
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    all_rows = []
    replay_count = 0
    for path in iter_replay_paths(args.patterns):
        replay = load_replay(path)
        if replay is None:
            continue
        replay_count += 1
        all_rows.extend(rows_for_replay(path, replay, args.horizon))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        args.output.write_text("", encoding="utf-8")
        print("replays: 0")
        print("rows: 0")
        return
    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"replays: {replay_count}")
    print(f"rows: {len(all_rows)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
