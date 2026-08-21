#!/usr/bin/env python3
"""Utilities for Kaggle Lux AI 2021 episode replay datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PLAYER_IDS = (0, 1)
DAY_LENGTH = 30
DAY_NIGHT_LENGTH = 40


def is_night_turn(turn: int) -> bool:
    return turn % DAY_NIGHT_LENGTH >= DAY_LENGTH


def load_replay(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as replay_file:
        return json.load(replay_file)


def iter_replay_paths(raw_root: Path) -> Iterable[Path]:
    seen = set()
    for path in sorted(raw_root.glob("data/*.json")):
        seen.add(path.resolve())
        yield path
    for path in sorted(raw_root.glob("*/*.json")):
        if path.resolve() not in seen:
            yield path


def replay_submission_id(path: Path) -> str:
    return path.parent.name


def replay_episode_id(path: Path, replay: Dict[str, Any]) -> str:
    info = replay.get("info", {})
    episode_id = info.get("EpisodeId") or replay.get("id") or path.stem
    return str(episode_id)


def team_names(replay: Dict[str, Any]) -> List[str]:
    names = replay.get("info", {}).get("TeamNames") or []
    return [str(name) for name in names]


def final_rewards(replay: Dict[str, Any]) -> List[Optional[float]]:
    rewards = replay.get("rewards") or []
    parsed = []
    for reward in rewards[:2]:
        try:
            parsed.append(float(reward))
        except (TypeError, ValueError):
            parsed.append(None)
    while len(parsed) < 2:
        parsed.append(None)
    return parsed


def infer_rank(rewards: List[Optional[float]], player: int) -> Optional[int]:
    own = rewards[player] if player < len(rewards) else None
    other = rewards[1 - player] if 1 - player < len(rewards) else None
    if own is None or other is None:
        return None
    if own == other:
        return 1
    return 1 if own > other else 2


def observation_for_step(replay: Dict[str, Any], step_index: int, player: int = 0) -> Dict[str, Any]:
    return replay["steps"][step_index][player]["observation"]


def actions_for_step(replay: Dict[str, Any], step_index: int, player: int) -> List[str]:
    return list(replay["steps"][step_index][player].get("action") or [])


def parse_updates(updates: List[str]) -> Dict[str, Any]:
    metrics = {
        player: {
            "research": 0,
            "cities": 0,
            "city_tiles": 0,
            "units": 0,
            "workers": 0,
            "carts": 0,
            "fuel": 0.0,
            "upkeep": 0.0,
        }
        for player in PLAYER_IDS
    }
    city_ids = {player: set() for player in PLAYER_IDS}

    for update in updates:
        parts = update.split()
        if not parts:
            continue
        kind = parts[0]
        if kind == "rp" and len(parts) >= 3:
            player = int(parts[1])
            metrics[player]["research"] = int(float(parts[2]))
        elif kind == "u" and len(parts) >= 4:
            unit_type = int(parts[1])
            player = int(parts[2])
            metrics[player]["units"] += 1
            if unit_type == 0:
                metrics[player]["workers"] += 1
            elif unit_type == 1:
                metrics[player]["carts"] += 1
        elif kind == "c" and len(parts) >= 5:
            player = int(parts[1])
            city_ids[player].add(parts[2])
            metrics[player]["fuel"] += float(parts[3])
            metrics[player]["upkeep"] += float(parts[4])
        elif kind == "ct" and len(parts) >= 4:
            player = int(parts[1])
            city_ids[player].add(parts[2])
            metrics[player]["city_tiles"] += 1

    for player in PLAYER_IDS:
        metrics[player]["cities"] = len(city_ids[player])
    return metrics


def replay_turn_metrics(replay: Dict[str, Any]) -> List[Dict[int, Dict[str, Any]]]:
    rows = []
    for step_index, _step in enumerate(replay.get("steps") or []):
        obs = observation_for_step(replay, step_index, player=0)
        rows.append(parse_updates(list(obs.get("updates") or [])))
    return rows


def player_night_losses(turn_metrics: List[Dict[int, Dict[str, Any]]], player: int) -> List[Dict[str, int]]:
    losses = []
    previous = None
    for turn, metrics_by_player in enumerate(turn_metrics):
        current = int(metrics_by_player[player]["city_tiles"])
        if previous is not None and current < previous and is_night_turn(turn):
            losses.append(
                {
                    "turn": turn,
                    "lost": previous - current,
                    "before": previous,
                    "after": current,
                }
            )
        previous = current
    return losses
