#!/usr/bin/env python3
"""Analyze downloaded Kaggle Lux AI 2021 replay JSON files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from replay_dataset_utils import (
    PLAYER_IDS,
    final_rewards,
    infer_rank,
    iter_replay_paths,
    load_replay,
    player_night_losses,
    replay_episode_id,
    replay_submission_id,
    replay_turn_metrics,
    team_names,
)


def replay_rows(path: Path) -> List[Dict[str, Any]]:
    replay = load_replay(path)
    teams = team_names(replay)
    rewards = final_rewards(replay)
    turns = len(replay.get("steps") or [])
    first_obs = replay["steps"][0][0]["observation"]
    turn_metrics = replay_turn_metrics(replay)
    final_metrics = turn_metrics[-1]

    rows = []
    for player in PLAYER_IDS:
        losses = player_night_losses(turn_metrics, player)
        final = final_metrics[player]
        rows.append(
            {
                "submission_id": replay_submission_id(path),
                "episode_id": replay_episode_id(path, replay),
                "file": str(path),
                "player": player,
                "team_name": teams[player] if player < len(teams) else "",
                "opponent_name": teams[1 - player] if 1 - player < len(teams) else "",
                "width": int(first_obs.get("width", 0)),
                "height": int(first_obs.get("height", 0)),
                "turns": turns,
                "reward": rewards[player],
                "rank": infer_rank(rewards, player),
                "alive_at_end": int(final["city_tiles"] > 0),
                "city_tiles": final["city_tiles"],
                "cities": final["cities"],
                "units": final["units"],
                "workers": final["workers"],
                "carts": final["carts"],
                "research": final["research"],
                "fuel": round(final["fuel"], 3),
                "upkeep": round(final["upkeep"], 3),
                "total_night_city_loss": sum(loss["lost"] for loss in losses),
                "max_night_city_loss": max((loss["lost"] for loss in losses), default=0),
                "night_loss_turns": ";".join(
                    f"{loss['turn']}:{loss['lost']}" for loss in losses
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_aggregate(rows: List[Dict[str, Any]]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["submission_id"], row["team_name"])].append(row)

    print(f"replay files: {len(rows) // 2}")
    print(f"player rows: {len(rows)}")
    print()
    print("submission_id,team,games,win_rate,alive_rate,mean_city_tiles,mean_research,total_night_loss,max_night_loss")
    for (submission_id, team), group in sorted(grouped.items()):
        games = len(group)
        wins = sum(row["rank"] == 1 for row in group)
        alive = sum(row["alive_at_end"] for row in group)
        mean_tiles = sum(row["city_tiles"] for row in group) / games
        mean_research = sum(row["research"] for row in group) / games
        total_night_loss = sum(row["total_night_city_loss"] for row in group)
        max_night_loss = max(row["max_night_city_loss"] for row in group)
        print(
            f"{submission_id},{team},{games},{wins / games:.3f},{alive / games:.3f},"
            f"{mean_tiles:.2f},{mean_research:.1f},{total_night_loss},{max_night_loss}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Kaggle episode replays from dataset/raw."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/replay_summary.csv"),
    )
    args = parser.parse_args()

    rows = []
    for path in iter_replay_paths(args.raw_root):
        rows.extend(replay_rows(path))

    write_csv(args.output, rows)
    print_aggregate(rows)
    print(f"\nsummary: {args.output}")


if __name__ == "__main__":
    main()
