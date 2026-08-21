#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lux_ai.nns import create_model  # noqa: E402
from lux_ai.utils import flags_to_namespace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a shallow ResNet checkpoint into a deeper model.")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flags = flags_to_namespace(OmegaConf.to_container(OmegaConf.load(args.target_config)))
    model = create_model(flags, torch.device("cpu"))
    source = torch.load(args.source_checkpoint, map_location="cpu")["model_state_dict"]
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "progressive_source": str(args.source_checkpoint),
        "loaded_keys": len(compatible),
        "new_keys": list(incompatible.missing_keys),
    }, args.output)
    print(f"loaded={len(compatible)} new={len(incompatible.missing_keys)} output={args.output}")


if __name__ == "__main__":
    main()
