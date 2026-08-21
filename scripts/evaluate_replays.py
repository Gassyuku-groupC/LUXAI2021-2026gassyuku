#!/usr/bin/env python3
"""Summarize Lux AI 2021 stateful replays for automated model promotion."""

import argparse
import json
from pathlib import Path


def player_metrics(replay, player):
    states = replay["stateful"]
    final = states[-1]
    team = final["teamStates"][str(player)]
    ranks = {
        int(item["agentID"]): int(item["rank"])
        for item in replay.get("results", {}).get("ranks", [])
    }
    city_tiles_by_turn = []
    for state in states:
        tiles = sum(
            len(city["cityCells"])
            for city in state["cities"].values()
            if int(city["team"]) == player
        )
        city_tiles_by_turn.append((int(state["turn"]), tiles))
    night_drops = []
    for (_, previous), (turn, current) in zip(
        city_tiles_by_turn, city_tiles_by_turn[1:]
    ):
        if current < previous and turn % 40 >= 30:
            night_drops.append(
                {"turn": turn, "lost": previous - current, "before": previous, "after": current}
            )
    final_cities = [
        city for city in final["cities"].values() if int(city["team"]) == player
    ]
    return {
        "rank": ranks.get(player),
        "turns": int(final["turn"]),
        "alive_at_end": len(final_cities) > 0,
        "reached_turn_360": int(final["turn"]) >= 360,
        "effective_survival": len(final_cities) > 0 and (
            int(final["turn"]) >= 360 or ranks.get(player) == 1
        ),
        "survived_360": int(final["turn"]) >= 360 and len(final_cities) > 0,
        "cities": len(final_cities),
        "city_tiles": sum(len(city["cityCells"]) for city in final_cities),
        "units": len(team["units"]),
        "research": int(team["researchPoints"]),
        "fuel": sum(float(city["fuel"]) for city in final_cities),
        "upkeep": sum(float(city["lightupkeep"]) for city in final_cities),
        "max_night_city_loss": max((drop["lost"] for drop in night_drops), default=0),
        "night_city_losses": night_drops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument(
        "--player",
        type=int,
        choices=(0, 1),
        help="Evaluate this player in every replay. Defaults to filename suffix _p0/_p1.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    games = []
    for path in args.replays:
        with path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
        player = args.player
        if player is None:
            if "_p0" in path.stem:
                player = 0
            elif "_p1" in path.stem:
                player = 1
            else:
                raise ValueError(f"Cannot infer player from {path.name}; add _p0 or _p1")
        games.append({"file": str(path), "player": player, **player_metrics(replay, player)})
    summary = {
        "games": len(games),
        "wins": sum(game["rank"] == 1 for game in games),
        "win_rate": sum(game["rank"] == 1 for game in games) / len(games),
        "survival_rate": sum(game["survived_360"] for game in games) / len(games),
        "effective_survival_rate": sum(game["effective_survival"] for game in games) / len(games),
        "mean_city_tiles": sum(game["city_tiles"] for game in games) / len(games),
        "mean_research": sum(game["research"] for game in games) / len(games),
        "worst_night_city_loss": max(game["max_night_city_loss"] for game in games),
        "details": games,
    }
    rendered = json.dumps(summary, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
