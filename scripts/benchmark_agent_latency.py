#!/usr/bin/env python3
"""Measure Lux agent cold-start and per-turn latency on real replay states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import yaml


class Observation(dict):
    def __init__(self, payload: dict):
        super().__init__(payload)
        self.player = int(payload["player"])


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-turns", type=int, default=360)
    parser.add_argument("--device", default="package", help="package, cpu, or cuda:N")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    agent_dir = args.agent.resolve()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    records = [
        step[args.player]["observation"]
        for step in replay["steps"][: args.max_turns]
        if len(step) > args.player and step[args.player].get("observation")
    ]
    if not records:
        raise ValueError("Replay contains no observations for the selected player")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(agent_dir))
    process_start = time.perf_counter()
    import torch  # noqa: E402
    from lux_ai.rl_agent import rl_agent as agent_module  # noqa: E402

    temporary_config = None
    if args.device != "package":
        config = yaml.safe_load(agent_module.RL_AGENT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["device"] = args.device
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        yaml.safe_dump(config, handle, sort_keys=False)
        handle.close()
        temporary_config = Path(handle.name)
        agent_module.RL_AGENT_CONFIG_PATH = temporary_config

    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    configuration = dict(replay.get("configuration") or {})
    if int(configuration.get("width", -1)) <= 0:
        configuration["width"] = int(records[0]["width"])
    if int(configuration.get("height", -1)) <= 0:
        configuration["height"] = int(records[0]["height"])
    first = Observation(records[0])
    model_agent = agent_module.RLAgent(first, configuration)
    synchronize()
    model_agent(first, configuration)
    synchronize()
    cold_start_seconds = time.perf_counter() - process_start

    latencies = []
    for payload in records[1:]:
        observation = Observation(payload)
        synchronize()
        started = time.perf_counter()
        model_agent(observation, configuration)
        synchronize()
        latencies.append(time.perf_counter() - started)

    per_turn_overage = [max(value - 3.0, 0.0) for value in latencies]
    cold_overage = max(cold_start_seconds - 3.0, 0.0)
    total_overage = cold_overage + sum(per_turn_overage)
    result = {
        "agent": str(agent_dir),
        "replay": str(args.replay.resolve()),
        "player": args.player,
        "device": args.device,
        "turns_measured": 1 + len(latencies),
        "cold_start_seconds": cold_start_seconds,
        "turn_mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "turn_p50_seconds": percentile(latencies, 0.50),
        "turn_p95_seconds": percentile(latencies, 0.95),
        "turn_max_seconds": max(latencies, default=0.0),
        "turns_over_3_seconds": sum(value > 3.0 for value in latencies),
        "estimated_overage_seconds": total_overage,
        "within_60_second_overage": total_overage < 60.0,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if temporary_config is not None:
        temporary_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
