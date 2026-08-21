#!/usr/bin/env python3
"""Rebuild an Excel-friendly index from downloaded replay JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def replay_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as replay_file:
        replay = json.load(replay_file)
    first_obs = replay["steps"][0][0]["observation"]
    return {
        "episode_id": replay.get("info", {}).get("EpisodeId") or path.stem,
        "path": str(path),
        "bytes": path.stat().st_size,
        "teams": " | ".join(str(name) for name in replay.get("info", {}).get("TeamNames") or []),
        "width": first_obs.get("width", ""),
        "height": first_obs.get("height", ""),
        "turns": len(replay.get("steps") or []),
        "rewards": " | ".join(str(reward) for reward in replay.get("rewards") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Index downloaded replay JSON files.")
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/raw/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/raw/downloaded_replays_index_excel.csv"),
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        try:
            rows.append(replay_row(path))
        except Exception as exc:
            rows.append(
                {
                    "episode_id": path.stem,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "teams": "",
                    "width": "",
                    "height": "",
                    "turns": "",
                    "rewards": f"error:{exc.__class__.__name__}",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=["episode_id", "path", "bytes", "teams", "width", "height", "turns", "rewards"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"indexed {len(rows)} files: {args.output}")


if __name__ == "__main__":
    main()
