#!/usr/bin/env python3
"""Verify exact Step-0 equivalence of the external spatial sidecar wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as replay_file:
        return json.load(replay_file)


def replay_size(replay: dict) -> int:
    observation = replay["steps"][0][0]["observation"]
    width = int(observation["width"])
    height = int(observation["height"])
    if width != height:
        raise ValueError(f"Only square maps are supported, got {width}x{height}")
    return width


def find_replays(root: Path, map_sizes: list[int]) -> Dict[int, Path]:
    wanted = set(map_sizes)
    found: Dict[int, Path] = {}
    for replay_path in sorted(root.rglob("*.json")):
        try:
            replay = load_json(replay_path)
            size = replay_size(replay)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if size in wanted and size not in found:
            found[size] = replay_path
            if len(found) == len(wanted):
                break
    missing = sorted(wanted.difference(found))
    if missing:
        raise FileNotFoundError(f"No valid replay found for map sizes: {missing}")
    return found


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def model_input_from_replay(replay: dict, flags, requested_step: int) -> tuple[dict, int]:
    steps = replay["steps"]
    target_step = min(max(int(requested_step), 0), len(steps) - 1)
    first_updates = steps[0][0]["observation"]["updates"]
    env = build_manual_env(flags, first_updates)
    placeholder = action_placeholder(env)
    for step_index in range(1, target_step + 1):
        updates = steps[step_index][0]["observation"].get("updates")
        if not updates:
            target_step = step_index - 1
            break
        advance_manual_env(env, step_index, updates)
    return env_output_for_current_state(env, placeholder), target_step


def finite_max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    finite = torch.isfinite(left) & torch.isfinite(right)
    if not finite.any():
        return 0.0
    return float((left[finite] - right[finite]).abs().max().item())


def compare_outputs(base_output: dict, wrapper_output: dict) -> dict:
    spaces = sorted(base_output["policy_logits"])
    per_space = {}
    action_equal = 0
    action_total = 0
    max_logit_diff = 0.0
    max_delta = 0.0
    for action_space in spaces:
        base_logits = base_output["policy_logits"][action_space]
        final_logits = wrapper_output["policy_logits"][action_space]
        base_actions = base_output["actions"][action_space]
        final_actions = wrapper_output["actions"][action_space]
        mask_equal = torch.equal(torch.isfinite(base_logits), torch.isfinite(final_logits))
        logits_exact = torch.equal(base_logits, final_logits)
        logits_diff = finite_max_diff(base_logits, final_logits)
        equal_count = int((base_actions == final_actions).sum().item())
        total_count = int(base_actions.numel())
        delta = wrapper_output["logit_deltas"].get(action_space)
        delta_abs = 0.0 if delta is None else float(delta.abs().max().item())
        per_space[action_space] = {
            "mask_exact": mask_equal,
            "logits_exact": logits_exact,
            "max_finite_logit_diff": logits_diff,
            "actions_equal": equal_count,
            "actions_total": total_count,
            "action_agreement": equal_count / max(total_count, 1),
            "max_abs_logit_delta": delta_abs,
        }
        action_equal += equal_count
        action_total += total_count
        max_logit_diff = max(max_logit_diff, logits_diff)
        max_delta = max(max_delta, delta_abs)

    baseline_diff = finite_max_diff(base_output["baseline"], wrapper_output["baseline"])
    return {
        "action_spaces": per_space,
        "max_finite_logit_diff": max_logit_diff,
        "max_abs_logit_delta": max_delta,
        "baseline_exact": torch.equal(base_output["baseline"], wrapper_output["baseline"]),
        "max_baseline_diff": baseline_diff,
        "actions_equal": action_equal,
        "actions_total": action_total,
        "action_agreement": action_equal / max(action_total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=Path("outputs/submission_packages/best_agent"),
    )
    parser.add_argument("--replay-root", type=Path, default=Path("dataset/raw/data"))
    parser.add_argument("--map-sizes", default="12,16,24,32")
    parser.add_argument("--step", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/spatial_risk_sidecar_step0/step0_equivalence.json"),
    )
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    map_sizes = [int(part.strip()) for part in args.map_sizes.split(",") if part.strip()]
    device = torch.device(args.device)
    flags = load_agent_flags(args.agent_dir)
    checkpoint_path = args.agent_dir / "lux_ai" / "rl_agent" / "candidate_weights.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = checkpoint["model_state_dict"]
    base_agent = create_model(flags, device).to(device)
    incompatible = base_agent.load_state_dict(checkpoint_state, strict=True)
    base_agent.eval()

    checkpoint_hash = state_dict_sha256(checkpoint_state)
    base_hash_before = state_dict_sha256(base_agent.state_dict())
    base_keys_before = tuple(base_agent.state_dict().keys())
    sidecar = SpatialRiskAttentionSidecar(
        in_channels=int(base_agent.base_out_channels),
        attention_dim=args.attention_dim,
        num_heads=args.num_heads,
        pool_size=args.pool_size,
    ).to(device)
    gate = SidecarLogitDeltaGate().to(device)
    wrapper = SidecarAgentWrapper(base_agent, sidecar, gate).to(device).eval()
    replay_paths = find_replays(args.replay_root, map_sizes)

    map_results = {}
    overall_equal = 0
    overall_total = 0
    for map_size in map_sizes:
        replay_path = replay_paths[map_size]
        replay = load_json(replay_path)
        model_input, actual_step = model_input_from_replay(replay, flags, args.step)
        model_input = to_device(model_input, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            base_output = base_agent.select_best_actions(model_input, return_features=True)
            wrapper_output = wrapper.select_best_actions(
                model_input,
                return_features=True,
                return_sidecar_outputs=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        comparison = compare_outputs(base_output, wrapper_output)
        shapes = wrapper_output["logical_shapes"].detach().cpu().tolist()
        comparison.update({
            "replay": str(replay_path.resolve()),
            "step": actual_step,
            "logical_shapes": shapes,
            "risk_tensor_shape": list(wrapper_output["risk_logits"].shape),
            "elapsed_seconds_base_plus_wrapper": time.perf_counter() - started,
        })
        expected_shapes = [[[map_size, map_size], [map_size, map_size]]]
        comparison["logical_shape_exact"] = shapes == expected_shapes
        map_results[str(map_size)] = comparison
        overall_equal += comparison["actions_equal"]
        overall_total += comparison["actions_total"]

    wrapper.assert_base_unchanged()
    base_hash_after = state_dict_sha256(base_agent.state_dict())
    base_keys_after = tuple(base_agent.state_dict().keys())
    max_diff = max(result["max_finite_logit_diff"] for result in map_results.values())
    max_baseline_diff = max(result["max_baseline_diff"] for result in map_results.values())
    max_delta = max(result["max_abs_logit_delta"] for result in map_results.values())
    all_masks_exact = all(
        space["mask_exact"]
        for result in map_results.values()
        for space in result["action_spaces"].values()
    )
    all_logits_exact = all(
        space["logits_exact"]
        for result in map_results.values()
        for space in result["action_spaces"].values()
    )
    passed = (
        checkpoint_hash == base_hash_before == base_hash_after
        and base_keys_before == base_keys_after
        and not incompatible.missing_keys
        and not incompatible.unexpected_keys
        and gate.zero_projection_is_exact()
        and all(not parameter.requires_grad for parameter in base_agent.parameters())
        and all(result["logical_shape_exact"] for result in map_results.values())
        and all_masks_exact
        and all_logits_exact
        and max_diff <= args.tolerance
        and max_baseline_diff <= args.tolerance
        and max_delta == 0.0
        and overall_equal == overall_total
    )
    report = {
        "passed": passed,
        "device": str(device),
        "tolerance": args.tolerance,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_hash": checkpoint_hash,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_state_keys_exact": base_keys_before == base_keys_after,
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "base_parameters_frozen": all(not parameter.requires_grad for parameter in base_agent.parameters()),
        "zero_projection_exact": gate.zero_projection_is_exact(),
        "max_finite_logit_diff": max_diff,
        "max_baseline_diff": max_baseline_diff,
        "max_abs_logit_delta": max_delta,
        "actions_equal": overall_equal,
        "actions_total": overall_total,
        "action_agreement": overall_equal / max(overall_total, 1),
        "maps": map_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
