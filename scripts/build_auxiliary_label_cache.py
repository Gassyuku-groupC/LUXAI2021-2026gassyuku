#!/usr/bin/env python3
"""Cache auxiliary risk labels for imitation shards."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@lru_cache(maxsize=128)
def load_replay_timeline(path_text: str) -> dict:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as replay_file:
        replay = json.load(replay_file)
    steps = replay.get("steps") or []
    city_counts = []
    unsafe_large = []
    for step in steps:
        updates = []
        if step and step[0].get("observation"):
            updates = step[0]["observation"].get("updates") or []
        counts = [0, 0]
        city_fuel = [{}, {}]
        city_upkeep = [{}, {}]
        city_tiles = [{}, {}]
        for update in updates:
            parts = update.split()
            if not parts:
                continue
            if parts[0] == "ct" and len(parts) >= 4:
                team = int(parts[1])
                city_id = parts[2]
                counts[team] += 1
                city_tiles[team][city_id] = city_tiles[team].get(city_id, 0) + 1
            elif parts[0] == "c" and len(parts) >= 5:
                team = int(parts[1])
                city_id = parts[2]
                city_fuel[team][city_id] = float(parts[3])
                city_upkeep[team][city_id] = float(parts[4])
        city_counts.append(counts)
        unsafe_large.append([
            any(
                tiles >= 20
                and city_fuel[team].get(city_id, 0.0) / max(city_upkeep[team].get(city_id, 1.0), 1.0) < 12.0
                for city_id, tiles in city_tiles[team].items()
            )
            for team in (0, 1)
        ])
    return {"city_counts": city_counts, "unsafe_large": unsafe_large}


def labels_for_meta(meta: Iterable[dict]) -> dict:
    players = []
    loss10 = []
    loss20 = []
    unsafe_large_city = []
    files = []
    state_steps = []
    for item in meta:
        file_text = str(item["file"])
        timeline = load_replay_timeline(file_text)
        step = int(item["state_step"])
        player = int(item["teacher_player"])
        counts = timeline["city_counts"]
        current = counts[min(step, len(counts) - 1)][player] if counts else 0
        future10 = counts[step + 1: min(step + 11, len(counts))]
        future20 = counts[step + 1: min(step + 21, len(counts))]
        min10 = min([row[player] for row in future10], default=current)
        min20 = min([row[player] for row in future20], default=current)
        unsafe = (
            timeline["unsafe_large"][min(step, len(timeline["unsafe_large"]) - 1)][player]
            if timeline["unsafe_large"]
            else False
        )
        players.append(player)
        loss10.append(float(min10 < current))
        loss20.append(float(min20 < current))
        unsafe_large_city.append(float(unsafe))
        files.append(file_text)
        state_steps.append(step)
    return {
        "players": torch.tensor(players, dtype=torch.long),
        "loss10": torch.tensor(loss10, dtype=torch.float32),
        "loss20": torch.tensor(loss20, dtype=torch.float32),
        "unsafe_large_city": torch.tensor(unsafe_large_city, dtype=torch.float32),
        "file": files,
        "state_step": torch.tensor(state_steps, dtype=torch.long),
    }


def summarize(labels: dict) -> dict:
    count = int(labels["players"].numel())
    return {
        "count": count,
        "loss10_positive_rate": float(labels["loss10"].mean().item()) if count else 0.0,
        "loss20_positive_rate": float(labels["loss20"].mean().item()) if count else 0.0,
        "unsafe_large_city_positive_rate": float(labels["unsafe_large_city"].mean().item()) if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached auxiliary labels for shard files.")
    parser.add_argument("--shards-dir", type=Path, default=Path("dataset/processed/imitation_shards_counterfactual_v4_residual"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/processed/auxiliary_labels_counterfactual_v4_residual"))
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(args.shards_dir.glob("shard_*.pt"))
    if args.max_shards:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise ValueError(f"No shard files found in {args.shards_dir}")

    all_summaries = []
    for shard_path in shard_paths:
        out_path = args.output_dir / shard_path.name
        if out_path.exists() and not args.overwrite:
            labels = torch.load(out_path, map_location="cpu")
            row = {"shard": shard_path.name, "cached": True, **summarize(labels)}
            all_summaries.append(row)
            print(row)
            continue
        shard = torch.load(shard_path, map_location="cpu")
        labels = labels_for_meta(shard["meta"])
        labels["source_shard"] = shard_path.name
        torch.save(labels, out_path)
        row = {"shard": shard_path.name, "cached": False, **summarize(labels)}
        all_summaries.append(row)
        print(row)

    total = sum(row["count"] for row in all_summaries)
    summary = {
        "shards_dir": str(args.shards_dir),
        "output_dir": str(args.output_dir),
        "shards": all_summaries,
        "total": total,
        "loss10_positive_rate": (
            sum(row["loss10_positive_rate"] * row["count"] for row in all_summaries) / max(total, 1)
        ),
        "loss20_positive_rate": (
            sum(row["loss20_positive_rate"] * row["count"] for row in all_summaries) / max(total, 1)
        ),
        "unsafe_large_city_positive_rate": (
            sum(row["unsafe_large_city_positive_rate"] * row["count"] for row in all_summaries) / max(total, 1)
        ),
    }
    summary_path = args.output_dir / "label_cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
