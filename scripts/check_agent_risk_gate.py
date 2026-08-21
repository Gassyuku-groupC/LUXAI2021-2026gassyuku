#!/usr/bin/env python3
"""Inspect whether a copied Lux agent has the experimental risk gate enabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Check risk-gate settings in an agent package.")
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--fail-on-block", action="store_true")
    args = parser.parse_args()

    config_path = args.agent_dir / "lux_ai" / "rl_agent" / "rl_agent_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    result = {
        "agent_dir": str(args.agent_dir),
        "risk_gate_enabled": bool(config.get("risk_gate_enabled", False)),
        "risk_gate_mode": str(config.get("risk_gate_mode", "unset")),
        "risk_gate_max_turn": config.get("risk_gate_max_turn"),
        "risk_gate_bw_fuel_turns_lt": config.get("risk_gate_bw_fuel_turns_lt"),
        "fuel_support_enabled": bool(config.get("fuel_support_enabled", False)),
        "fuel_support_mode": str(config.get("fuel_support_mode", "unset")),
        "fuel_support_max_turn": config.get("fuel_support_max_turn"),
        "fuel_support_city_fuel_turns_lt": config.get("fuel_support_city_fuel_turns_lt"),
        "fuel_support_min_cargo_fuel": config.get("fuel_support_min_cargo_fuel"),
    }
    print(json.dumps(result, indent=2))
    if args.fail_on_block and result["risk_gate_enabled"] and result["risk_gate_mode"] == "block":
        raise SystemExit("risk gate is in block mode")


if __name__ == "__main__":
    main()
