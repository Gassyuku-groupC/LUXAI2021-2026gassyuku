#!/usr/bin/env python3
"""Extract per-turn strategy features from Lux replay files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple


DAY_LEN = 30
DN_CYCLE_LEN = 40
PLAYER_IDS = (0, 1)


def iter_replay_paths(patterns: list[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for raw_path in glob.glob(pattern, recursive=True):
            path = Path(raw_path)
            if path.name.endswith(".commands.json") or path.name.endswith(".log"):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def state_from_updates(updates: list[str], width: int, height: int) -> dict:
    state = {
        "map": [[{} for _ in range(width)] for _ in range(height)],
        "cities": {},
        "teamStates": {
            "0": {"researchPoints": 0, "units": {}, "researched": {}},
            "1": {"researchPoints": 0, "units": {}, "researched": {}},
        },
    }
    for update in updates:
        parts = str(update).split()
        if not parts:
            continue
        kind = parts[0]
        if kind == "rp" and len(parts) >= 3:
            team = int(parts[1])
            state["teamStates"].setdefault(str(team), {"units": {}})["researchPoints"] = int(float(parts[2]))
        elif kind == "r" and len(parts) >= 5:
            resource_type, x, y, amount = parts[1], int(parts[2]), int(parts[3]), int(float(parts[4]))
            if 0 <= y < height and 0 <= x < width:
                state["map"][y][x]["resource"] = {"type": resource_type, "amount": amount}
        elif kind == "u" and len(parts) >= 10:
            unit_type, team, unit_id = int(parts[1]), int(parts[2]), parts[3]
            x, y = int(parts[4]), int(parts[5])
            cooldown = float(parts[6])
            wood, coal, uranium = int(float(parts[7])), int(float(parts[8])), int(float(parts[9]))
            state["teamStates"].setdefault(str(team), {"researchPoints": 0, "units": {}})["units"][unit_id] = {
                "type": unit_type,
                "team": team,
                "id": unit_id,
                "x": x,
                "y": y,
                "cooldown": cooldown,
                "cargo": {"wood": wood, "coal": coal, "uranium": uranium},
            }
        elif kind == "c" and len(parts) >= 5:
            team, city_id = int(parts[1]), parts[2]
            state["cities"][city_id] = {
                "team": team,
                "cityid": city_id,
                "fuel": float(parts[3]),
                "lightupkeep": float(parts[4]),
                "cityCells": [],
            }
        elif kind == "ct" and len(parts) >= 6:
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


def load_replay_bundle(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as replay_file:
            raw = json.load(replay_file)
    except Exception:
        return None

    if isinstance(raw, dict) and isinstance(raw.get("steps"), list):
        return kaggle_bundle(path, raw)
    if isinstance(raw, dict) and isinstance(raw.get("stateful"), list):
        return stateful_bundle(path, raw)
    return None


def kaggle_bundle(path: Path, raw: dict) -> Optional[dict]:
    steps = raw.get("steps") or []
    if not steps:
        return None
    first_obs = (steps[0][0] or {}).get("observation") or {}
    width = int(first_obs.get("width") or 0)
    height = int(first_obs.get("height") or 0)
    states = []
    for turn_steps in steps:
        obs = (turn_steps[0] or {}).get("observation") or {}
        states.append(state_from_updates(list(obs.get("updates") or []), width, height))

    commands = [[] for _ in states]
    # Kaggle steps[i].action is the action emitted for state i - 1.
    for action_step in range(1, len(steps)):
        state_turn = action_step - 1
        for player, agent_step in enumerate(steps[action_step][:2]):
            for action in agent_step.get("action") or []:
                commands[state_turn].append({"agentID": player, "command": action})

    names = [str(name) for name in raw.get("info", {}).get("TeamNames") or []]
    while len(names) < 2:
        names.append("")
    rewards = []
    for value in (raw.get("rewards") or [])[:2]:
        try:
            rewards.append(float(value))
        except (TypeError, ValueError):
            rewards.append(None)
    while len(rewards) < 2:
        rewards.append(None)

    return {
        "source_format": "kaggle",
        "file": str(path),
        "episode_id": str(raw.get("info", {}).get("EpisodeId") or raw.get("id") or path.stem),
        "submission_id": path.parent.name,
        "states": states,
        "commands": commands,
        "width": width,
        "height": height,
        "team_names": names,
        "rewards": rewards,
        "ranks": infer_ranks(rewards),
    }


def stateful_bundle(path: Path, raw: dict) -> Optional[dict]:
    states = raw.get("stateful") or []
    if not states:
        return None
    commands = raw.get("allCommands") or [[] for _ in states]
    while len(commands) < len(states):
        commands.append([])
    team_details = raw.get("teamDetails") or []
    names = [str(item.get("name", "")) for item in team_details[:2] if isinstance(item, dict)]
    while len(names) < 2:
        names.append("")
    ranks = [None, None]
    for item in (raw.get("results") or {}).get("ranks") or []:
        agent = int(item.get("agentID", -1))
        if agent in PLAYER_IDS:
            ranks[agent] = int(item.get("rank", 0) or 0)
    return {
        "source_format": "stateful",
        "file": str(path),
        "episode_id": str(raw.get("seed") or path.stem),
        "submission_id": path.parent.name,
        "states": states,
        "commands": commands,
        "width": int(raw.get("width") or len(states[0].get("map") or [])),
        "height": int(raw.get("height") or len(states[0].get("map") or [])),
        "team_names": names,
        "rewards": [None, None],
        "ranks": ranks,
    }


def infer_ranks(rewards: list[Optional[float]]) -> list[Optional[int]]:
    if len(rewards) < 2 or rewards[0] is None or rewards[1] is None:
        return [None, None]
    if rewards[0] == rewards[1]:
        return [1, 1]
    return [1, 2] if rewards[0] > rewards[1] else [2, 1]


def team_state(state: dict, team: int) -> dict:
    return state.get("teamStates", {}).get(str(team), {}) or {}


def team_units(state: dict, team: int) -> dict:
    return team_state(state, team).get("units", {}) or {}


def team_cities(state: dict, team: int) -> list[tuple[str, dict]]:
    return [
        (city_id, city)
        for city_id, city in (state.get("cities") or {}).items()
        if int(city.get("team", -1)) == team
    ]


def city_positions(city: dict) -> set[Tuple[int, int]]:
    return {(int(cell["x"]), int(cell["y"])) for cell in city.get("cityCells") or []}


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def city_at(state: dict, team: int, x: int, y: int) -> Optional[tuple[str, dict]]:
    for city_id, city in team_cities(state, team):
        if any(int(cell["x"]) == x and int(cell["y"]) == y for cell in city.get("cityCells") or []):
            return city_id, city
    return None


def adjacent_cities(state: dict, team: int, pos: Tuple[int, int]) -> list[tuple[str, dict]]:
    found = []
    for city_id, city in team_cities(state, team):
        if any(manhattan(pos, city_pos) == 1 for city_pos in city_positions(city)):
            found.append((city_id, city))
    return found


def near_resource(state: dict, pos: Tuple[int, int], radius: int = 2) -> bool:
    game_map = state.get("map") or []
    height = len(game_map)
    width = len(game_map[0]) if height else 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            x, y = pos[0] + dx, pos[1] + dy
            if 0 <= x < width and 0 <= y < height:
                resource = (game_map[y][x] or {}).get("resource")
                if resource and int(resource.get("amount", 0)) > 0:
                    return True
    return False


def fuel_turns(city: dict) -> float:
    return float(city.get("fuel", 0.0)) / max(float(city.get("lightupkeep", 1.0)), 1.0)


def resource_amounts(state: dict) -> dict[str, int]:
    totals = {"wood": 0, "coal": 0, "uranium": 0}
    for row in state.get("map") or []:
        for cell in row:
            resource = (cell or {}).get("resource")
            if resource:
                totals[str(resource.get("type", ""))] = totals.get(str(resource.get("type", "")), 0) + int(resource.get("amount", 0))
    return totals


def future_team_loss(states: list[dict], turn: int, team: int, horizon: int) -> int:
    max_dt = min(horizon, len(states) - turn - 1)
    if max_dt <= 0:
        return 0
    now = city_tile_count(states[turn], team)
    worst = 0
    for dt in range(1, max_dt + 1):
        worst = max(worst, now - city_tile_count(states[turn + dt], team))
    return worst


def city_tile_count(state: dict, team: int) -> int:
    return sum(len(city.get("cityCells") or []) for _, city in team_cities(state, team))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[int(idx)]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def unit_fuel(unit: dict) -> int:
    cargo = unit.get("cargo", {}) or {}
    return int(cargo.get("wood", 0)) + int(cargo.get("coal", 0)) * 10 + int(cargo.get("uranium", 0)) * 40


def action_features(state: dict, commands: list[dict], team: int) -> dict:
    features = {
        "action_count": 0,
        "move_actions": 0,
        "transfer_actions": 0,
        "pillage_actions": 0,
        "research_actions": 0,
        "bw_actions": 0,
        "bc_actions": 0,
        "bcity_actions": 0,
        "bcity_isolated_actions": 0,
        "bcity_adjacent_actions": 0,
        "bcity_resource_near_actions": 0,
        "bcity_adjacent_low_fuel_lt5_actions": 0,
        "bw_low_fuel_lt3_actions": 0,
        "bw_low_fuel_lt5_actions": 0,
        "bw_low_fuel_lt10_actions": 0,
    }
    for command in commands:
        if int(command.get("agentID", -1)) != team:
            continue
        raw = str(command.get("command", ""))
        parts = raw.split()
        if not parts:
            continue
        features["action_count"] += 1
        kind = parts[0]
        if kind == "m":
            features["move_actions"] += 1
        elif kind == "t":
            features["transfer_actions"] += 1
        elif kind == "p":
            features["pillage_actions"] += 1
        elif kind == "r":
            features["research_actions"] += 1
        elif kind == "bc":
            features["bc_actions"] += 1
        elif kind == "bw":
            features["bw_actions"] += 1
            if len(parts) >= 3:
                try:
                    target = city_at(state, team, int(parts[1]), int(parts[2]))
                except ValueError:
                    target = None
                if target:
                    value = fuel_turns(target[1])
                    if value < 3:
                        features["bw_low_fuel_lt3_actions"] += 1
                    if value < 5:
                        features["bw_low_fuel_lt5_actions"] += 1
                    if value < 10:
                        features["bw_low_fuel_lt10_actions"] += 1
        elif kind == "bcity":
            features["bcity_actions"] += 1
            if len(parts) >= 2:
                unit = team_units(state, team).get(parts[1])
                if unit:
                    pos = (int(unit["x"]), int(unit["y"]))
                    adjacent = adjacent_cities(state, team, pos)
                    if adjacent:
                        features["bcity_adjacent_actions"] += 1
                        if min(fuel_turns(city) for _, city in adjacent) < 5:
                            features["bcity_adjacent_low_fuel_lt5_actions"] += 1
                    else:
                        features["bcity_isolated_actions"] += 1
                    if near_resource(state, pos):
                        features["bcity_resource_near_actions"] += 1
    return features


def city_features(state: dict, team: int) -> dict:
    cities = team_cities(state, team)
    units = team_units(state, team)
    city_sizes = [len(city.get("cityCells") or []) for _, city in cities]
    fuel_values = [fuel_turns(city) for _, city in cities]
    total_fuel = sum(float(city.get("fuel", 0.0)) for _, city in cities)
    total_upkeep = sum(float(city.get("lightupkeep", 0.0)) for _, city in cities)
    workers = sum(1 for unit in units.values() if int(unit.get("type", 0)) == 0)
    carts = sum(1 for unit in units.values() if int(unit.get("type", 0)) == 1)
    city_tiles = sum(city_sizes)
    resource_near_cities = 0
    isolated_cities = 0
    for city_id, city in cities:
        positions = city_positions(city)
        if any(near_resource(state, pos) for pos in positions):
            resource_near_cities += 1
        other_positions = [
            other_pos
            for other_id, other_city in cities
            if other_id != city_id
            for other_pos in city_positions(other_city)
        ]
        if positions and not any(manhattan(pos, other_pos) <= 3 for pos in positions for other_pos in other_positions):
            isolated_cities += 1
    return {
        "cities": len(cities),
        "city_tiles": city_tiles,
        "largest_city_size": max(city_sizes, default=0),
        "mean_city_size": (sum(city_sizes) / len(city_sizes)) if city_sizes else 0.0,
        "resource_near_cities": resource_near_cities,
        "isolated_cities_r3": isolated_cities,
        "units": len(units),
        "workers": workers,
        "carts": carts,
        "unit_cap_margin": city_tiles - len(units),
        "worker_citytile_ratio": workers / max(city_tiles, 1),
        "research": int(team_state(state, team).get("researchPoints", 0) or 0),
        "fuel": total_fuel,
        "upkeep": total_upkeep,
        "fuel_turns_total": total_fuel / max(total_upkeep, 1.0),
        "min_city_fuel_turns": min(fuel_values, default=0.0),
        "p25_city_fuel_turns": percentile(fuel_values, 0.25),
        "median_city_fuel_turns": percentile(fuel_values, 0.50),
        "mean_city_fuel_turns": (sum(fuel_values) / len(fuel_values)) if fuel_values else 0.0,
        "low_fuel_city_lt3": sum(value < 3 for value in fuel_values),
        "low_fuel_city_lt5": sum(value < 5 for value in fuel_values),
        "low_fuel_city_lt10": sum(value < 10 for value in fuel_values),
        "unit_cargo_fuel": sum(unit_fuel(unit) for unit in units.values()),
    }


def rows_for_bundle(bundle: dict, horizons: list[int]) -> list[dict]:
    rows = []
    states = bundle["states"]
    commands_by_turn = bundle["commands"]
    final_turn = len(states) - 1
    final_tiles = {team: city_tile_count(states[-1], team) for team in PLAYER_IDS}
    final_units = {team: len(team_units(states[-1], team)) for team in PLAYER_IDS}
    for turn, state in enumerate(states):
        resources = resource_amounts(state)
        cycle_turn = turn % DN_CYCLE_LEN
        phase = "night" if cycle_turn >= DAY_LEN else "pre_night" if cycle_turn >= DAY_LEN - 5 else "day"
        for team in PLAYER_IDS:
            opponent = 1 - team
            row = {
                "file": bundle["file"],
                "source_format": bundle["source_format"],
                "submission_id": bundle["submission_id"],
                "episode_id": bundle["episode_id"],
                "team": team,
                "team_name": bundle["team_names"][team] if team < len(bundle["team_names"]) else "",
                "opponent_name": bundle["team_names"][opponent] if opponent < len(bundle["team_names"]) else "",
                "rank": bundle["ranks"][team] if team < len(bundle["ranks"]) else "",
                "reward": bundle["rewards"][team] if team < len(bundle["rewards"]) else "",
                "map_size": bundle["width"],
                "width": bundle["width"],
                "height": bundle["height"],
                "turn": turn,
                "turns_remaining": max(final_turn - turn, 0),
                "cycle_turn": cycle_turn,
                "phase": phase,
                "pre_night": int(phase == "pre_night"),
                "is_night": int(phase == "night"),
                "turns_to_night": 0 if cycle_turn >= DAY_LEN else DAY_LEN - cycle_turn,
                "wood_remaining": resources.get("wood", 0),
                "coal_remaining": resources.get("coal", 0),
                "uranium_remaining": resources.get("uranium", 0),
                "final_city_tiles": final_tiles[team],
                "final_units": final_units[team],
            }
            row.update(city_features(state, team))
            row.update(action_features(state, commands_by_turn[turn] if turn < len(commands_by_turn) else [], team))
            for horizon in horizons:
                row[f"future_team_loss_{horizon}"] = future_team_loss(states, turn, team, horizon)
            if turn + 1 < len(states):
                row["city_tiles_delta_next"] = city_tile_count(states[turn + 1], team) - city_tile_count(state, team)
                row["units_delta_next"] = len(team_units(states[turn + 1], team)) - len(team_units(state, team))
                row["research_delta_next"] = (
                    int(team_state(states[turn + 1], team).get("researchPoints", 0) or 0)
                    - int(team_state(state, team).get("researchPoints", 0) or 0)
                )
            else:
                row["city_tiles_delta_next"] = 0
                row["units_delta_next"] = 0
                row["research_delta_next"] = 0
            rows.append(row)
    return rows


def parse_horizons(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_team_filter(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-turn strategy features from Lux replays.")
    parser.add_argument("patterns", nargs="+", help="Replay glob patterns, e.g. dataset/raw/**/*.json")
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/strategy_features.csv"))
    parser.add_argument("--horizons", default="1,3,5,10")
    parser.add_argument("--team-names", default="", help="Comma-separated team names to keep. Empty keeps both teams.")
    parser.add_argument("--map-sizes", default="", help="Comma-separated map sizes to keep. Empty keeps all.")
    parser.add_argument("--max-replays", type=int, default=0)
    args = parser.parse_args()

    team_filter = parse_team_filter(args.team_names)
    map_filter = {int(part.strip()) for part in args.map_sizes.split(",") if part.strip()}
    horizons = parse_horizons(args.horizons)
    all_rows = []
    replay_count = 0
    skipped = 0
    for path in iter_replay_paths(args.patterns):
        bundle = load_replay_bundle(path)
        if bundle is None:
            skipped += 1
            continue
        if map_filter and int(bundle["width"]) not in map_filter:
            continue
        if team_filter and not any(name in team_filter for name in bundle["team_names"]):
            continue
        replay_rows = rows_for_bundle(bundle, horizons)
        if team_filter:
            replay_rows = [row for row in replay_rows if row["team_name"] in team_filter]
        all_rows.extend(replay_rows)
        replay_count += 1
        if args.max_replays and replay_count >= args.max_replays:
            break

    write_csv(args.output, all_rows)
    print(f"replays: {replay_count}")
    print(f"skipped: {skipped}")
    print(f"rows: {len(all_rows)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
