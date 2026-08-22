#!/usr/bin/env python3
"""Verify best and zero-delta sidecar equivalence on every replay state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imitation_learning_utils import (  # noqa: E402
    action_placeholder,
    advance_manual_env,
    build_manual_env,
    env_output_for_current_state,
    load_agent_flags,
)
from lux_ai.nns import create_model  # noqa: E402
from lux_ai.rl_agent.learned_intervention_gate import SidecarLogitDeltaGate  # noqa: E402
from lux_ai.rl_agent.sidecar_agent_wrapper import SidecarAgentWrapper, state_dict_sha256  # noqa: E402
from lux_ai.rl_agent.spatial_risk_sidecar import SpatialRiskAttentionSidecar  # noqa: E402
from verify_step0_equivalence import compare_outputs, find_replays, load_json, to_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--wrapper-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--map-sizes", default="12,16,24,32")
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    flags = load_agent_flags(args.agent_dir)
    base_checkpoint = torch.load(
        args.agent_dir / "lux_ai" / "rl_agent" / "candidate_weights.pt",
        map_location=device,
        weights_only=False,
    )["model_state_dict"]
    base = create_model(flags, device).to(device)
    base.load_state_dict(base_checkpoint, strict=True)
    base.eval()
    sidecar = SpatialRiskAttentionSidecar(int(base.base_out_channels), 64, 4, 8).to(device)
    gate = SidecarLogitDeltaGate().to(device)
    wrapper = SidecarAgentWrapper(base, sidecar, gate).to(device).eval()
    wrapper_state = torch.load(
        args.wrapper_checkpoint,
        map_location=device,
        weights_only=False,
    )["model_state_dict"]
    wrapper.load_state_dict(wrapper_state, strict=True)
    wrapper._base_state_hash = state_dict_sha256(wrapper.base_agent.state_dict())

    sizes = [int(value) for value in args.map_sizes.split(",")]
    replay_paths = find_replays(args.replay_root, sizes)
    results = {}
    total_actions = 0
    total_equal = 0
    max_logit_diff = 0.0
    max_baseline_diff = 0.0
    max_delta = 0.0

    for size in sizes:
        replay = load_json(replay_paths[size])
        steps = replay["steps"][: args.max_steps]
        env = build_manual_env(flags, steps[0][0]["observation"]["updates"])
        placeholder = action_placeholder(env)
        checked = 0
        for index, step in enumerate(steps):
            if index:
                updates = step[0]["observation"].get("updates")
                if not updates:
                    break
                advance_manual_env(env, index, updates)
            model_input = to_device(
                env_output_for_current_state(env, placeholder),
                device,
            )
            with torch.no_grad():
                base_output = base.select_best_actions(model_input, return_features=True)
                wrapper_output = wrapper.select_best_actions(
                    model_input,
                    return_features=True,
                    return_sidecar_outputs=True,
                )
            comparison = compare_outputs(base_output, wrapper_output)
            if (
                comparison["max_finite_logit_diff"] != 0.0
                or comparison["max_baseline_diff"] != 0.0
                or comparison["max_abs_logit_delta"] != 0.0
                or comparison["actions_equal"] != comparison["actions_total"]
            ):
                raise AssertionError(f"Equivalence failed at map={size} step={index}: {comparison}")
            checked += 1
            total_actions += comparison["actions_total"]
            total_equal += comparison["actions_equal"]
            max_logit_diff = max(max_logit_diff, comparison["max_finite_logit_diff"])
            max_baseline_diff = max(max_baseline_diff, comparison["max_baseline_diff"])
            max_delta = max(max_delta, comparison["max_abs_logit_delta"])
        results[str(size)] = {
            "replay": str(replay_paths[size].resolve()),
            "states_checked": checked,
        }

    wrapper.assert_base_unchanged()
    report = {
        "passed": True,
        "base_state_hash": state_dict_sha256(base_checkpoint),
        "maps": results,
        "states_checked": sum(item["states_checked"] for item in results.values()),
        "actions_equal": total_equal,
        "actions_total": total_actions,
        "action_agreement": total_equal / max(total_actions, 1),
        "max_finite_logit_diff": max_logit_diff,
        "max_baseline_diff": max_baseline_diff,
        "max_abs_logit_delta": max_delta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
