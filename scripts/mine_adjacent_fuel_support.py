#!/usr/bin/env python3
"""Mine missed adjacent fuel-support opportunities from stateful Lux replays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_strategy_features import (  # noqa: E402
    DAY_LEN,
    DN_CYCLE_LEN,
    PLAYER_IDS,
    city_positions,
    city_tile_count,
    fuel_turns,
    future_team_loss,
    iter_replay_paths,
    load_replay_bundle,
    manhattan,
    team_cities,
    team_units,
    unit_fuel,
)


DIRECTIONS = {
    "n": (0, -1),
    "s": (0, 1),
    "e": (1, 0),
    "w": (-1, 0),
    "c": (0, 0),
}


def parse_eval_side(path: str) -> int | None:
    match = re.search(r"_p([01])(?:\.json)?$", os.path.basename(path))
    return int(match.group(1)) if match else None


def command_by_unit(commands: list[dict], team: int) -> dict[str, str]:
    out = {}
    for command in commands:
        if int(command.get("agentID", -1)) != team:
            continue
        raw = str(command.get("command", ""))
        parts = raw.split()
        if len(parts) >= 2 and parts[0] in {"m", "bcity", "p"}:
            out[parts[1]] = raw
        elif len(parts) >= 5 and parts[0] == "t":
            out[parts[1]] = raw
    return out


def move_target(unit: dict, action: str) -> tuple[int, int] | None:
    parts = str(action).split()
    if len(parts) != 3 or parts[0] != "m":
        return None
    dx, dy = DIRECTIONS.get(parts[2], (0, 0))
    return int(unit.get("x", 0)) + dx, int(unit.get("y", 0)) + dy


def is_own_city_tile(state: dict, team: int, pos: tuple[int, int]) -> bool:
    for _, city in team_cities(state, team):
        if pos in city_positions(city):
            return True
    return False


def low_fuel_city_cells(state: dict, team: int, threshold: float) -> list[dict]:
    rows = []
    for city_id, city in team_cities(state, team):
        turns = fuel_turns(city)
        if turns >= threshold:
            continue
        for cell in city.get("cityCells") or []:
            rows.append(
                {
                    "city_id": city_id,
                    "city_fuel_turns": turns,
                    "city_fuel": float(city.get("fuel", 0.0)),
                    "city_upkeep": float(city.get("lightupkeep", 0.0)),
                    "city_tiles": len(city.get("cityCells") or []),
                    "x": int(cell["x"]),
                    "y": int(cell["y"]),
                }
            )
    return rows


def candidate_teams(bundle: dict, args: argparse.Namespace) -> Iterable[int]:
    if args.all_teams:
        return PLAYER_IDS
    side = parse_eval_side(bundle["file"])
    if side is None:
        return PLAYER_IDS
    return [side]


def mine_bundle(bundle: dict, args: argparse.Namespace) -> list[dict]:
    rows = []
    states = bundle["states"]
    commands_by_turn = bundle["commands"]
    final_tiles = {team: city_tile_count(states[-1], team) for team in PLAYER_IDS}
    for turn, state in enumerate(states):
        cycle_turn = turn % DN_CYCLE_LEN
        is_night = cycle_turn >= DAY_LEN
        turns_to_night = 0 if is_night else DAY_LEN - cycle_turn
        if not is_night and turns_to_night > args.turns_to_night_lte:
            continue
        if is_night and not args.include_night:
            continue
        if turn > args.max_turn:
            continue

        commands = commands_by_turn[turn] if turn < len(commands_by_turn) else []
        for team in candidate_teams(bundle, args):
            low_cells = low_fuel_city_cells(state, team, args.city_fuel_turns_lt)
            if not low_cells:
                continue
            actions = command_by_unit(commands, team)
            for unit_id, unit in team_units(state, team).items():
                if int(unit.get("type", -1)) != 0:
                    continue
                if float(unit.get("cooldown", 0.0)) >= 1.0:
                    continue
                unit_pos = (int(unit.get("x", 0)), int(unit.get("y", 0)))
                if is_own_city_tile(state, team, unit_pos):
                    continue
                cargo_fuel = unit_fuel(unit)
                if cargo_fuel < args.min_cargo_fuel:
                    continue

                adjacent = [
                    cell for cell in low_cells
                    if manhattan(unit_pos, (cell["x"], cell["y"])) == 1
                ]
                if not adjacent:
                    continue
                adjacent.sort(key=lambda cell: (cell["city_fuel_turns"], cell["city_tiles"]))
                target_cell = adjacent[0]
                action = actions.get(unit_id, "")
                target = move_target(unit, action) if action else None
                supporting = target == (target_cell["x"], target_cell["y"])
                if supporting and not args.include_supporting:
                    continue
                event_type = "supporting_adjacent_fuel" if supporting else "missed_adjacent_fuel"
                rows.append(
                    {
                        "event_type": event_type,
                        "file": bundle["file"],
                        "episode_id": bundle["episode_id"],
                        "team": team,
                        "team_name": bundle["team_names"][team] if team < len(bundle["team_names"]) else "",
                        "opponent_name": bundle["team_names"][1 - team] if 1 - team < len(bundle["team_names"]) else "",
                        "rank": bundle["ranks"][team] if team < len(bundle["ranks"]) else "",
                        "map_size": bundle["width"],
                        "turn": turn,
                        "cycle_turn": cycle_turn,
                        "turns_to_night": turns_to_night,
                        "is_night": int(is_night),
                        "unit_id": unit_id,
                        "unit_x": unit_pos[0],
                        "unit_y": unit_pos[1],
                        "unit_cargo_fuel": cargo_fuel,
                        "action": action,
                        "move_target_x": target[0] if target else "",
                        "move_target_y": target[1] if target else "",
                        "city_id": target_cell["city_id"],
                        "city_x": target_cell["x"],
                        "city_y": target_cell["y"],
                        "city_fuel_turns": target_cell["city_fuel_turns"],
                        "city_fuel": target_cell["city_fuel"],
                        "city_upkeep": target_cell["city_upkeep"],
                        "city_tiles": target_cell["city_tiles"],
                        "team_city_tiles": city_tile_count(state, team),
                        "final_city_tiles": final_tiles[team],
                        "future_team_loss_5": future_team_loss(states, turn, team, 5),
                        "future_team_loss_10": future_team_loss(states, turn, team, 10),
                        "future_team_loss_20": future_team_loss(states, turn, team, 20),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(json.dumps({"rows": 0}, indent=2) + "\n", encoding="utf-8")
        return
    data = pd.DataFrame(rows)
    data["loss_10"] = pd.to_numeric(data["future_team_loss_10"], errors="coerce").fillna(0) > 0
    data["big_loss_10"] = pd.to_numeric(data["future_team_loss_10"], errors="coerce").fillna(0) >= 5
    summary = {
        "rows": int(len(data)),
        "missed_rows": int((data["event_type"] == "missed_adjacent_fuel").sum()),
        "supporting_rows": int((data["event_type"] == "supporting_adjacent_fuel").sum()),
        "future_loss_10_rate": float(data["loss_10"].mean()),
        "future_big_loss_10_rate": float(data["big_loss_10"].mean()),
        "mean_future_loss_10": float(pd.to_numeric(data["future_team_loss_10"], errors="coerce").fillna(0).mean()),
        "by_event_type": (
            data.groupby("event_type")
            .agg(
                rows=("event_type", "size"),
                future_loss_10_rate=("loss_10", "mean"),
                future_big_loss_10_rate=("big_loss_10", "mean"),
                mean_future_loss_10=("future_team_loss_10", "mean"),
                mean_city_fuel_turns=("city_fuel_turns", "mean"),
                mean_unit_cargo_fuel=("unit_cargo_fuel", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table_path = path.with_suffix(".csv")
    data.groupby(["event_type", "team", "is_night"]).agg(
        rows=("event_type", "size"),
        future_loss_10_rate=("loss_10", "mean"),
        mean_future_loss_10=("future_team_loss_10", "mean"),
        mean_city_fuel_turns=("city_fuel_turns", "mean"),
    ).reset_index().to_csv(table_path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine adjacent fuel support opportunities from Lux stateful replays.")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--city-fuel-turns-lt", type=float, default=5.0)
    parser.add_argument("--min-cargo-fuel", type=float, default=20.0)
    parser.add_argument("--turns-to-night-lte", type=int, default=3)
    parser.add_argument("--max-turn", type=int, default=200)
    parser.add_argument("--include-night", action="store_true", default=True)
    parser.add_argument("--exclude-night", dest="include_night", action="store_false")
    parser.add_argument("--include-supporting", action="store_true")
    parser.add_argument("--all-teams", action="store_true")
    parser.add_argument("--max-replays", type=int, default=0)
    args = parser.parse_args()

    rows = []
    replay_count = 0
    for path in iter_replay_paths(args.patterns):
        bundle = load_replay_bundle(path)
        if bundle is None:
            continue
        rows.extend(mine_bundle(bundle, args))
        replay_count += 1
        if args.max_replays and replay_count >= args.max_replays:
            break

    write_csv(args.output, rows)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    write_summary(summary_path, rows)
    print(json.dumps({"replays": replay_count, "rows": len(rows), "output": str(args.output), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
