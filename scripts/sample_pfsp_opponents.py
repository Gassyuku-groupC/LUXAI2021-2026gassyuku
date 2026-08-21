#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lux_ai.torchbeast.pfsp import PFSPOpponentSampler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a PFSP league schedule.")
    parser.add_argument("--pool-config", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sampler = PFSPOpponentSampler.from_json(args.pool_config, seed=args.seed)
    schedule = [sampler.sample().name for _ in range(args.count)]
    payload = {"sampler": sampler.to_dict(), "schedule": schedule}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
