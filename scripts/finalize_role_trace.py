#!/usr/bin/env python3
"""Convert one player's runtime role JSONL trace into a replay sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.trace.is_file():
        raise FileNotFoundError(f"Role trace is missing: {args.trace}")
    replay = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    frames_by_turn = {}
    for line_number, line in enumerate(args.trace.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid role trace JSON at line {line_number}") from exc
        frames_by_turn[int(frame["turn"])] = frame

    frames = [frames_by_turn[turn] for turn in sorted(frames_by_turn)]
    payload = {
        "schema": "lux-role-overlay/v1",
        "seed": replay.get("seed"),
        "width": replay.get("width"),
        "height": replay.get("height"),
        "player": frames[0]["player"] if frames else None,
        "replay_file": args.replay.name,
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(frames)} role frames: {args.output}")


if __name__ == "__main__":
    main()
