#!/usr/bin/env python3
"""Create a side/phase index that emphasizes safe expansion, not only risk avoidance."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_imitation_index_risk_aware import (  # noqa: E402
    future_team_loss,
    grouped_rows,
    is_pre_night_window,
    is_risk_window,
    min_city_fuel_turns,
    parse_int_set,
    reweight_row,
    state_for_step,
)


def turn_bucket(turn: int) -> str:
    if turn < 40:
        return "000-039"
    if turn < 80:
        return "040-079"
    if turn < 120:
        return "080-119"
    if turn < 160:
        return "120-159"
    if turn < 240:
        return "160-239"
    if turn < 320:
        return "240-319"
    return "320-360"


def side_label(teacher_player: int) -> str:
    return f"p{teacher_player}"


def load_modifiers(path: Path) -> dict:
    if not path.exists():
        return {"side_phase": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def severity_scales(severity: int, args: argparse.Namespace) -> tuple[float, float]:
    if severity >= 3:
        return args.failure_safe_scale_severe, args.failure_risk_scale_severe
    if severity == 2:
        return args.failure_safe_scale_high, args.failure_risk_scale_high
    if severity == 1:
        return args.failure_safe_scale_moderate, args.failure_risk_scale_moderate
    return 1.0, 1.0


def city_growth(row: dict) -> int:
    before = int(float(row.get("city_tiles_before", 0) or 0))
    after = int(float(row.get("city_tiles_after", 0) or 0))
    return max(after - before, 0)


def apply_side_phase_and_expansion_feedback(
    row: dict,
    replay: dict,
    state_cache: dict[int, dict],
    modifiers: dict,
    args: argparse.Namespace,
) -> dict:
    row = dict(row)
    original_weight = float(row["weight"])
    state_step = int(row["state_step"])
    teacher_player = int(row["teacher_player"])
    key = f"{side_label(teacher_player)}:{turn_bucket(state_step)}"
    modifier = (modifiers.get("side_phase") or {}).get(key)
    severity = int(modifier.get("severity", 0)) if modifier else 0
    safe_scale, risk_scale = severity_scales(severity, args)

    state = state_for_step(replay, row, state_step, state_cache)
    min_fuel = min_city_fuel_turns(state, teacher_player)
    future_loss = future_team_loss(replay, row, state_cache, args.future_loss_horizon)
    night_loss_next = int(float(row.get("night_city_loss_next", 0) or 0))
    growth = city_growth(row)
    safe_buffer = min_fuel is not None and min_fuel >= args.safe_min_fuel_turns
    expansion_buffer = min_fuel is not None and min_fuel >= args.safe_expansion_min_fuel_turns
    no_near_loss = future_loss == 0 and night_loss_next == 0
    is_safe_example = no_near_loss and safe_buffer
    is_safe_expansion = no_near_loss and expansion_buffer and growth > 0
    is_risky_example = (
        future_loss > 0
        or night_loss_next > 0
        or (min_fuel is not None and min_fuel < args.risky_min_fuel_turns)
    )

    scale = 1.0
    reasons = [row.get("weight_reason", "")]
    if modifier:
        if is_safe_example:
            scale *= safe_scale
            reasons.append(f"{key}_safe_phase_x{safe_scale:g}")
            if is_pre_night_window(state_step, args.risk_window_lead_turns):
                scale *= args.safe_pre_night_extra_scale
                reasons.append(f"{key}_safe_pre_night_x{args.safe_pre_night_extra_scale:g}")
        elif is_risky_example and is_risk_window(state_step, args.risk_window_lead_turns):
            scale *= risk_scale
            reasons.append(f"{key}_risk_phase_x{risk_scale:g}")
        elif severity >= 2 and safe_buffer:
            scale *= args.buffered_same_phase_scale
            reasons.append(f"{key}_buffered_same_phase_x{args.buffered_same_phase_scale:g}")

    if is_safe_expansion:
        expansion_scale = args.safe_expansion_scale
        if int(float(row.get("rank", 2) or 2)) == 1:
            expansion_scale *= args.winner_safe_expansion_extra_scale
        if int(float(row.get("max_teacher_night_loss", 999) or 999)) <= args.safe_expansion_max_teacher_night_loss:
            expansion_scale *= args.low_loss_teacher_expansion_extra_scale
        if is_pre_night_window(state_step, args.risk_window_lead_turns):
            expansion_scale *= args.pre_night_safe_expansion_extra_scale
        scale *= expansion_scale
        reasons.append(f"safe_expansion_x{expansion_scale:g}")

    if no_near_loss and expansion_buffer and growth == 0 and severity >= 2:
        scale *= args.safe_non_growth_phase_scale
        reasons.append(f"{key}_safe_non_growth_x{args.safe_non_growth_phase_scale:g}")

    new_weight = min(max(original_weight * scale, args.min_weight), args.max_weight)
    row["weight"] = f"{new_weight:.4f}"
    row["weight_reason"] = "+".join(part for part in reasons if part)
    row["side_phase_key"] = key
    row["side_phase_severity"] = str(severity)
    row["side_phase_scale"] = f"{(new_weight / original_weight if original_weight else 0.0):.4f}"
    row["safe_expansion"] = "1" if is_safe_expansion else "0"
    row["city_growth_next"] = str(growth)
    return row


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = [
        "original_weight",
        "risk_severity",
        "future_team_loss_10",
        "min_city_fuel_turns",
        "risk_scale",
        "side_phase_key",
        "side_phase_severity",
        "side_phase_scale",
        "safe_expansion",
        "city_growth_next",
    ]
    out_fields = [field for field in fieldnames if field not in extras] + extras
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a safe-expansion side/phase imitation index.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_hq.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_side_phase_v4.csv"))
    parser.add_argument("--modifiers", type=Path, default=Path("outputs/risk_feature_logs/side_phase_failure_modifiers_v3.json"))
    parser.add_argument("--map-sizes", default="16")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--risk-window-lead-turns", type=int, default=5)
    parser.add_argument("--future-loss-horizon", type=int, default=10)
    parser.add_argument("--safe-min-fuel-turns", type=float, default=8.0)
    parser.add_argument("--safe-expansion-min-fuel-turns", type=float, default=8.0)
    parser.add_argument("--risky-min-fuel-turns", type=float, default=3.0)
    parser.add_argument("--failure-safe-scale-moderate", type=float, default=1.08)
    parser.add_argument("--failure-safe-scale-high", type=float, default=1.18)
    parser.add_argument("--failure-safe-scale-severe", type=float, default=1.30)
    parser.add_argument("--failure-risk-scale-moderate", type=float, default=0.95)
    parser.add_argument("--failure-risk-scale-high", type=float, default=0.85)
    parser.add_argument("--failure-risk-scale-severe", type=float, default=0.75)
    parser.add_argument("--safe-pre-night-extra-scale", type=float, default=1.08)
    parser.add_argument("--buffered-same-phase-scale", type=float, default=1.05)
    parser.add_argument("--safe-expansion-scale", type=float, default=1.35)
    parser.add_argument("--winner-safe-expansion-extra-scale", type=float, default=1.10)
    parser.add_argument("--low-loss-teacher-expansion-extra-scale", type=float, default=1.10)
    parser.add_argument("--pre-night-safe-expansion-extra-scale", type=float, default=1.05)
    parser.add_argument("--safe-non-growth-phase-scale", type=float, default=1.03)
    parser.add_argument("--safe-expansion-max-teacher-night-loss", type=int, default=10)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=5.0)

    # Softer base risk-aware parameters than v3: keep risk signal, avoid crushing scale.
    parser.add_argument("--bw-severe-fuel-turns", type=float, default=3.0)
    parser.add_argument("--bw-high-fuel-turns", type=float, default=5.0)
    parser.add_argument("--bw-moderate-fuel-turns", type=float, default=10.0)
    parser.add_argument("--bcity-severe-adjacent-fuel-turns", type=float, default=3.0)
    parser.add_argument("--bcity-high-adjacent-fuel-turns", type=float, default=5.0)
    parser.add_argument("--severe-scale", type=float, default=0.45)
    parser.add_argument("--high-scale", type=float, default=0.65)
    parser.add_argument("--moderate-scale", type=float, default=0.85)
    parser.add_argument("--next-loss-scale", type=float, default=0.75)
    parser.add_argument("--safe-risk-window-scale", type=float, default=1.15)
    parser.add_argument("--safe-pre-night-min-fuel-turns", type=float, default=8.0)
    parser.add_argument("--safe-pre-night-buffer-scale", type=float, default=1.15)
    parser.add_argument("--pre-night-bw-severe-extra-scale", type=float, default=0.85)
    parser.add_argument("--resource-backup-city-scale", type=float, default=1.15)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as in_file:
        fieldnames = csv.DictReader(in_file).fieldnames or []

    modifiers = load_modifiers(args.modifiers)
    grouped = grouped_rows(args.input, parse_int_set(args.map_sizes), args.max_rows)
    rows: list[dict] = []
    severity_counts: dict[int, int] = defaultdict(int)
    changed = 0
    safe_expansion_count = 0
    for replay_path, replay_rows in grouped.items():
        with replay_path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
        state_cache: dict[int, dict] = {}
        for base_row in replay_rows:
            risk_row = reweight_row(base_row, replay, state_cache, args)
            final_row = apply_side_phase_and_expansion_feedback(risk_row, replay, state_cache, modifiers, args)
            severity = int(final_row.get("side_phase_severity", 0) or 0)
            severity_counts[severity] += 1
            if float(final_row.get("side_phase_scale", 1.0)) != 1.0:
                changed += 1
            if final_row.get("safe_expansion") == "1":
                safe_expansion_count += 1
            rows.append(final_row)

    write_rows(args.output, rows, fieldnames)
    mean_weight = sum(float(row["weight"]) for row in rows) / len(rows) if rows else 0.0
    print(f"rows: {len(rows)}")
    print(f"mean weight: {mean_weight:.3f}" if rows else "mean weight: n/a")
    print(f"side-phase changed rows: {changed}")
    print(f"safe expansion rows: {safe_expansion_count}")
    print("side-phase severity:", dict(sorted(severity_counts.items())))
    print(f"index: {args.output}")


if __name__ == "__main__":
    main()
