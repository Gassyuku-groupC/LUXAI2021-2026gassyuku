#!/usr/bin/env python3
"""Build a weighted behavior-cloning index from Kaggle Lux replay JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from replay_dataset_utils import (
    final_rewards,
    infer_rank,
    is_night_turn,
    iter_replay_paths,
    load_replay,
    player_night_losses,
    replay_episode_id,
    replay_submission_id,
    replay_turn_metrics,
    team_names,
)


def load_manifest(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open(encoding="utf-8") as manifest_file:
        raw = json.load(manifest_file)
    entries = raw.get("teachers", raw if isinstance(raw, list) else [])
    by_submission: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        submission_id = str(entry.get("submission_id", "*"))
        by_submission.setdefault(submission_id, []).append(entry)
    return by_submission


def resolve_teacher_player(entry: Dict[str, Any], names: List[str]) -> Optional[int]:
    if "player" in entry:
        player = int(entry["player"])
        return player if player in (0, 1) else None
    team_name = str(entry.get("team_name", ""))
    for player, name in enumerate(names[:2]):
        if name == team_name:
            return player
    return None


def row_weight(
    *,
    state_turn: int,
    rank: Optional[int],
    before_tiles: int,
    after_tiles: int,
    final_tiles: int,
    drop_loss_turns: bool,
) -> tuple[float, str]:
    reasons = ["base"]
    weight = 1.0
    lost = max(before_tiles - after_tiles, 0)

    if rank == 1:
        weight += 0.25
        reasons.append("winner")
    if state_turn >= 240:
        weight += 0.25
        reasons.append("late")
    if is_night_turn(state_turn):
        weight += 0.50
        reasons.append("night")
        if before_tiles > 0 and lost == 0:
            weight += 0.75
            reasons.append("night_kept_city")
    if final_tiles > 0:
        weight += 0.25
        reasons.append("alive_end")
    if lost > 0:
        if drop_loss_turns:
            return 0.0, "dropped_city"
        weight *= 0.25
        reasons.append("downweighted_city_loss")

    return round(weight, 4), "+".join(reasons)


def replay_index_rows(
    path: Path,
    manifest_entries: List[Dict[str, Any]],
    drop_loss_turns: bool,
    only_winners: bool,
    max_teacher_night_loss: Optional[int],
) -> List[Dict[str, Any]]:
    replay = load_replay(path)
    submission_id = replay_submission_id(path)
    episode_id = replay_episode_id(path, replay)
    names = team_names(replay)
    rewards = final_rewards(replay)
    turn_metrics = replay_turn_metrics(replay)
    first_obs = replay["steps"][0][0]["observation"]
    rows: List[Dict[str, Any]] = []

    for entry in manifest_entries:
        teacher_player = resolve_teacher_player(entry, names)
        if teacher_player is None:
            print(
                f"skip {path}: cannot find teacher {entry.get('team_name')} "
                f"in teams={names}"
            )
            continue

        losses = player_night_losses(turn_metrics, teacher_player)
        loss_by_turn = {loss["turn"]: loss["lost"] for loss in losses}
        max_night_loss = max((loss["lost"] for loss in losses), default=0)
        final_tiles = int(turn_metrics[-1][teacher_player]["city_tiles"])
        rank = infer_rank(rewards, teacher_player)
        if only_winners and rank != 1:
            continue
        if max_teacher_night_loss is not None and max_night_loss > max_teacher_night_loss:
            continue

        # In Kaggle episode JSON, steps[0] is the initial observation with no
        # action; steps[i].action is the action emitted for steps[i - 1].
        for action_step in range(1, len(replay.get("steps") or [])):
            state_turn = action_step - 1
            actions = replay["steps"][action_step][teacher_player].get("action") or []
            if not actions:
                continue
            before_tiles = int(turn_metrics[state_turn][teacher_player]["city_tiles"])
            after_tiles = int(turn_metrics[action_step][teacher_player]["city_tiles"])
            weight, reason = row_weight(
                state_turn=state_turn,
                rank=rank,
                before_tiles=before_tiles,
                after_tiles=after_tiles,
                final_tiles=final_tiles,
                drop_loss_turns=drop_loss_turns,
            )
            if weight <= 0.0:
                continue
            rows.append(
                {
                    "submission_id": submission_id,
                    "episode_id": episode_id,
                    "file": str(path),
                    "teacher_team": names[teacher_player] if teacher_player < len(names) else "",
                    "teacher_player": teacher_player,
                    "opponent_team": names[1 - teacher_player] if 1 - teacher_player < len(names) else "",
                    "state_step": state_turn,
                    "action_step": action_step,
                    "width": int(first_obs.get("width", 0)),
                    "height": int(first_obs.get("height", 0)),
                    "is_night": int(is_night_turn(state_turn)),
                    "action_count": len(actions),
                    "city_tiles_before": before_tiles,
                    "city_tiles_after": after_tiles,
                    "night_city_loss_next": loss_by_turn.get(action_step, 0),
                    "max_teacher_night_loss": max_night_loss,
                    "rank": rank,
                    "weight": weight,
                    "weight_reason": reason,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a weighted index for replay-based imitation learning."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/teachers.json"),
        help="JSON file listing submission_id/team_name teacher mappings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/imitation_index.csv"),
    )
    parser.add_argument(
        "--drop-loss-turns",
        action="store_true",
        help="Exclude actions immediately followed by city-tile loss.",
    )
    parser.add_argument(
        "--only-winners",
        action="store_true",
        help="Use only teacher players that won the replay.",
    )
    parser.add_argument(
        "--max-teacher-night-loss",
        type=int,
        help="Skip a teacher game if its max single night city-tile loss exceeds this value.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    rows = []
    for path in iter_replay_paths(args.raw_root):
        entries = [
            *manifest.get("*", []),
            *manifest.get(replay_submission_id(path), []),
        ]
        if entries:
            rows.extend(
                replay_index_rows(
                    path,
                    entries,
                    args.drop_loss_turns,
                    args.only_winners,
                    args.max_teacher_night_loss,
                )
            )

    write_csv(args.output, rows)
    print(f"teacher rows: {len(rows)}")
    if rows:
        total_weight = sum(float(row["weight"]) for row in rows)
        night_rows = sum(int(row["is_night"]) for row in rows)
        print(f"night rows: {night_rows}")
        print(f"mean weight: {total_weight / len(rows):.3f}")
    print(f"index: {args.output}")


if __name__ == "__main__":
    main()
