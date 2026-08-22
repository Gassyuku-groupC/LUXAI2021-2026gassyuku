#!/usr/bin/env python3
"""Build a zero-delta sidecar checkpoint around the unchanged best Actor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lux_ai.rl_agent.sidecar_agent_wrapper import state_dict_sha256


def model_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint lacks model_state_dict: {path}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--sidecar-donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    best_state = model_state(args.best)
    donor_state = model_state(args.sidecar_donor)
    if any(key.startswith("base_agent.") for key in best_state):
        raise ValueError("Expected an unwrapped best Actor checkpoint")

    combined = {f"base_agent.{key}": value.detach().clone() for key, value in best_state.items()}
    retained_prefixes = ("spatial_risk_sidecar.", "intervention_gate.")
    retained = {
        key: value.detach().clone()
        for key, value in donor_state.items()
        if key.startswith(retained_prefixes)
    }
    if not retained:
        raise ValueError("Sidecar donor contains no external sidecar parameters")
    combined.update(retained)

    zeroed = []
    for key, value in combined.items():
        if key.startswith("intervention_gate.delta_projections."):
            value.zero_()
            zeroed.append(key)
    if not zeroed:
        raise ValueError("No intervention-gate delta projections were found")

    nonfinite = [
        key for key, value in combined.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise FloatingPointError(f"Non-finite tensors: {nonfinite[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "best_actor_sidecar_zero_delta_v1",
            "model_state_dict": combined,
            "base_state_hash": state_dict_sha256(best_state),
            "gate_delta_projection_zeroed": sorted(zeroed),
            "sidecar_donor": str(args.sidecar_donor.resolve()),
        },
        args.output,
    )
    print(f"Saved {args.output}")
    print(f"best_keys={len(best_state)} combined_keys={len(combined)} zeroed={len(zeroed)}")
    print(f"base_state_hash={state_dict_sha256(best_state)}")


if __name__ == "__main__":
    main()
