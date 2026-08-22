#!/usr/bin/env python3
"""Export learned Role-only checkpoint scalars to the runtime adapter YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lux_ai.rl_agent.trainable_role_bias import ROLE_BIAS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    values = {}
    for name in ROLE_BIAS_NAMES:
        key = f"role_bias_layer.bias_params.{name}"
        if key not in state:
            raise KeyError(f"Role bias missing from checkpoint: {key}")
        value = state[key]
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite role bias: {name}")
        values[name] = float(value.detach().cpu())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump({"role_city_bias_params": values}, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Exported {len(values)} role biases: {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()
