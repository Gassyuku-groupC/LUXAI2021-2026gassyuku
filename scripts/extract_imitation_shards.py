#!/usr/bin/env python3
"""Extract replay imitation-learning tensors from a weighted index."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from imitation_learning_utils import (
    action_placeholder,
    advance_manual_env,
    build_manual_env,
    env_output_for_current_state,
    load_agent_flags,
    teacher_actions_to_mask,
)


def parse_int_set(text: str) -> set[int]:
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def load_index(path: Path, max_rows: int = 0, map_sizes: set[int] | None = None) -> Dict[Path, List[dict]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            if map_sizes and int(row["width"]) not in map_sizes:
                continue
            row["state_step"] = int(row["state_step"])
            row["action_step"] = int(row["action_step"])
            row["teacher_player"] = int(row["teacher_player"])
            row["weight"] = float(row["weight"])
            grouped[Path(row["file"])].append(row)
            if max_rows and sum(len(v) for v in grouped.values()) >= max_rows:
                break
    for rows in grouped.values():
        rows.sort(key=lambda item: item["state_step"])
    return dict(grouped)


def _append_sample(buffers: dict, env_output: dict, target: dict, meta: dict, weight: float) -> None:
    buffers["weights"].append(torch.tensor(weight, dtype=torch.float32))
    buffers["critical_mask"].append(torch.tensor(float(meta.get("critical_mask", 0.0)), dtype=torch.float32))
    buffers["counterfactual_scale"].append(torch.tensor(float(meta.get("counterfactual_scale", 1.0)), dtype=torch.float32))
    buffers["meta"].append(meta)
    for key, value in env_output["obs"].items():
        buffers["obs"][key].append(value.squeeze(0).cpu())
    for key, value in env_output["info"]["available_actions_mask"].items():
        buffers["available_actions_mask"][key].append(value.squeeze(0).cpu())
    buffers["input_mask"].append(env_output["info"]["input_mask"].squeeze(0).cpu())
    for key, value in target.items():
        buffers["actions_taken"][key].append(torch.from_numpy(value).cpu())


def _empty_buffers() -> dict:
    return {
        "obs": defaultdict(list),
        "available_actions_mask": defaultdict(list),
        "actions_taken": defaultdict(list),
        "input_mask": [],
        "weights": [],
        "critical_mask": [],
        "counterfactual_scale": [],
        "meta": [],
    }


def _flush(buffers: dict, output_dir: Path, shard_index: int) -> int:
    if not buffers["weights"]:
        return shard_index
    output_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        "obs": {key: torch.stack(values) for key, values in buffers["obs"].items()},
        "available_actions_mask": {
            key: torch.stack(values) for key, values in buffers["available_actions_mask"].items()
        },
        "actions_taken": {
            key: torch.stack(values).to(torch.bool) for key, values in buffers["actions_taken"].items()
        },
        "input_mask": torch.stack(buffers["input_mask"]),
        "weights": torch.stack(buffers["weights"]),
        "critical_mask": torch.stack(buffers["critical_mask"]),
        "counterfactual_scale": torch.stack(buffers["counterfactual_scale"]),
        "meta": buffers["meta"],
    }
    out_path = output_dir / f"shard_{shard_index:05d}.pt"
    torch.save(shard, out_path)
    print(f"wrote {out_path} samples={len(buffers['weights'])}")
    return shard_index + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract imitation-learning tensor shards.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("dataset/processed/imitation_index_hq.csv"),
    )
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed/imitation_shards_hq"),
    )
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--critical-min-scale-delta", type=float, default=0.02)
    parser.add_argument(
        "--map-sizes",
        default="",
        help="Comma-separated map sizes to keep, for example 12,16. Empty keeps all maps.",
    )
    args = parser.parse_args()

    flags = load_agent_flags(args.agent_dir)
    grouped = load_index(args.index, args.max_rows, parse_int_set(args.map_sizes))
    buffers = _empty_buffers()
    shard_index = 0
    total = 0

    for replay_path, rows in grouped.items():
        with replay_path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
        steps = replay.get("steps") or []
        if not steps:
            continue
        env = build_manual_env(flags, list(steps[0][0]["observation"].get("updates") or []))
        placeholder = action_placeholder(env)
        current_step = 0

        for row in rows:
            while current_step < row["state_step"]:
                current_step += 1
                advance_manual_env(
                    env,
                    current_step,
                    list(steps[current_step][0]["observation"].get("updates") or []),
                )
            env_output = env_output_for_current_state(env, placeholder)
            actions = steps[row["action_step"]][row["teacher_player"]].get("action") or []
            target = teacher_actions_to_mask(
                env.unwrapped[0].game_state,
                row["teacher_player"],
                actions,
            )
            if not any(value.any() for value in target.values()):
                continue
            _append_sample(
                buffers,
                env_output,
                target,
                {
                    **row,
                    "file": str(replay_path),
                    "episode_id": row["episode_id"],
                    "state_step": row["state_step"],
                    "action_step": row["action_step"],
                    "teacher_player": row["teacher_player"],
                    "teacher_team": row["teacher_team"],
                    "counterfactual_scale": float(row.get("counterfactual_scale", 1.0) or 1.0),
                    "weight_reason": row.get("weight_reason", ""),
                    "critical_mask": float(
                        abs(float(row.get("counterfactual_scale", 1.0) or 1.0) - 1.0)
                        >= args.critical_min_scale_delta
                    ),
                },
                row["weight"],
            )
            total += 1
            if len(buffers["weights"]) >= args.shard_size:
                shard_index = _flush(buffers, args.output_dir, shard_index)
                buffers = _empty_buffers()

    shard_index = _flush(buffers, args.output_dir, shard_index)
    print(f"done: samples={total} shards={shard_index} output={args.output_dir}")


if __name__ == "__main__":
    main()
