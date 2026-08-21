#!/usr/bin/env python3
"""Counterfactual v3 imitation reweighting.

V3 keeps v2's fuel-support benefit, but backs away from broad macro
conservatism. It avoids penalizing bw-only rows, makes late penalties narrower,
and adds a small positive signal to safe worker/support behavior so the learner
does not solve risk by simply becoming small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_imitation_index_counterfactual_v2 as v2


def exact_scale(labels: list[dict], flags: dict[str, int], args: argparse.Namespace) -> tuple[float, list[str], int, int]:
    penalties = [row for row in labels if v2.as_int(row, "penalty_label")]
    safe_late = [row for row in labels if str(row.get("counterfactual_label", "")) == "late_risk_without_big_loss"]
    scale = 1.0
    reasons: list[str] = []

    if penalties:
        max_weight = max(v2.as_float(row, "penalty_weight") for row in penalties)
        risky_exact_action = flags["has_bcity"] or flags["has_research"]
        if risky_exact_action:
            factor = max(args.exact_min_scale, 1.0 - args.exact_penalty_strength * min(max_weight / 3.0, 1.0))
            scale *= factor
            reasons.append(f"counterfactual_v3_exact_penalty_x{factor:.3f}")

    if safe_late and not penalties:
        safe_action = flags["has_bw"] or flags["has_transfer"] or v2.city_growth(labels[0]) > 0
        if safe_action:
            factor = min(args.exact_safe_max_scale, 1.0 + args.exact_safe_strength)
            scale *= factor
            reasons.append(f"counterfactual_v3_exact_safe_x{factor:.3f}")

    if not reasons:
        reasons.append("counterfactual_v3_exact_neutral")
    return scale, reasons, len(penalties), len(safe_late)


def phase_scale(row: dict, flags: dict[str, int], profile: dict, args: argparse.Namespace) -> tuple[float, list[str], int, int]:
    turn = v2.as_int(row, "state_step")
    team = v2.as_int(row, "teacher_player")
    max_teacher_night_loss = v2.as_int(row, "max_teacher_night_loss")
    night_loss_next = v2.as_int(row, "night_city_loss_next")
    growth = v2.city_growth(row)
    safe_teacher = max_teacher_night_loss <= args.safe_teacher_max_night_loss and night_loss_next <= 0
    safe_boost_teacher = max_teacher_night_loss <= args.safe_boost_max_night_loss and night_loss_next <= 0

    scale = 1.0
    reasons: list[str] = []
    penalty_hits = 0
    safe_hits = 0

    late_item = profile.get(v2.phase_key(team, turn, "late_big_loss_warning"))
    if late_item:
        has_late_penalty_profile = (
            int(late_item["penalty"]) >= args.min_phase_penalties
            and float(late_item["penalty_rate"]) >= args.min_phase_penalty_rate
        )
        has_late_safe_profile = int(late_item["safe_late"]) >= args.min_safe_phase_rows

        if has_late_penalty_profile:
            phase_strength = min(float(late_item["mean_penalty_weight"]) / 3.0, 1.0)
            risky_late_action = flags["has_bcity"] or (turn >= args.start_turn and flags["has_research"])
            inefficient_growth = growth > 0 and max_teacher_night_loss > args.safe_teacher_max_night_loss
            if risky_late_action or inefficient_growth:
                penalty_hits += 1
                factor = max(args.phase_min_scale, 1.0 - args.phase_late_strength * phase_strength)
                scale *= factor
                reasons.append(f"counterfactual_v3_late_penalty_x{factor:.3f}")

        if has_late_safe_profile and safe_boost_teacher and turn >= args.start_turn:
            safe_development = flags["has_transfer"] or flags["has_bw"] or growth > 0
            if safe_development:
                safe_hits += 1
                factor = 1.0 + args.safe_late_strength
                if flags["has_bw"]:
                    factor += args.safe_bw_extra
                if growth > 0:
                    factor += args.safe_growth_extra
                factor = min(args.safe_phase_max_scale, factor)
                scale *= factor
                reasons.append(f"counterfactual_v3_late_safe_x{factor:.3f}")

    support_item = profile.get(v2.phase_key(team, turn, "suggest_fuel_support"))
    if support_item:
        has_support_penalty_profile = (
            int(support_item["penalty"]) >= args.min_phase_penalties
            and float(support_item["penalty_rate"]) >= args.min_phase_penalty_rate
        )
        if has_support_penalty_profile:
            phase_strength = min(float(support_item["mean_penalty_weight"]) / 3.0, 1.0)
            unsupported_window = (
                v2.risk_window(turn, args.risk_window_lead_turns)
                and not flags["has_transfer"]
                and (flags["has_bcity"] or flags["has_research"])
            )
            if unsupported_window:
                penalty_hits += 1
                factor = max(args.support_phase_min_scale, 1.0 - args.phase_support_strength * phase_strength)
                scale *= factor
                reasons.append(f"counterfactual_v3_support_penalty_x{factor:.3f}")

        if (
            int(support_item["support_neutral"]) >= args.min_safe_phase_rows
            and safe_teacher
            and v2.risk_window(turn, args.risk_window_lead_turns)
            and (flags["has_transfer"] or flags["has_bw"])
        ):
            safe_hits += 1
            factor = 1.0 + args.safe_support_strength
            if flags["has_bw"]:
                factor += args.safe_bw_extra
            factor = min(args.safe_phase_max_scale, factor)
            scale *= factor
            reasons.append(f"counterfactual_v3_support_safe_x{factor:.3f}")

    if not reasons:
        reasons.append("counterfactual_v3_no_action_match")
    return scale, reasons, penalty_hits, safe_hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply counterfactual v3 scale-preserving risk reweighting.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_hq.csv"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_risk_labels_v1_from_best/counterfactual_risk_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_counterfactual_v3.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_reweight_v3/summary.json"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--start-turn", type=int, default=140)
    parser.add_argument("--enable-phase-fallback", action="store_true", default=True)
    parser.add_argument("--min-phase-penalties", type=int, default=2)
    parser.add_argument("--min-phase-penalty-rate", type=float, default=0.60)
    parser.add_argument("--min-safe-phase-rows", type=int, default=1)
    parser.add_argument("--safe-teacher-max-night-loss", type=int, default=10)
    parser.add_argument("--safe-boost-max-night-loss", type=int, default=8)
    parser.add_argument("--risk-window-lead-turns", type=int, default=5)
    parser.add_argument("--exact-penalty-strength", type=float, default=0.28)
    parser.add_argument("--exact-min-scale", type=float, default=0.70)
    parser.add_argument("--exact-safe-strength", type=float, default=0.10)
    parser.add_argument("--exact-safe-max-scale", type=float, default=1.16)
    parser.add_argument("--phase-late-strength", type=float, default=0.10)
    parser.add_argument("--phase-support-strength", type=float, default=0.08)
    parser.add_argument("--phase-min-scale", type=float, default=0.86)
    parser.add_argument("--support-phase-min-scale", type=float, default=0.90)
    parser.add_argument("--safe-late-strength", type=float, default=0.06)
    parser.add_argument("--safe-growth-extra", type=float, default=0.02)
    parser.add_argument("--safe-support-strength", type=float, default=0.05)
    parser.add_argument("--safe-bw-extra", type=float, default=0.04)
    parser.add_argument("--safe-phase-max-scale", type=float, default=1.16)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=5.0)
    args = parser.parse_args()

    v2.exact_scale = exact_scale
    v2.phase_scale = phase_scale
    v2.process_index(args)
    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        summary["version"] = "counterfactual_v3"
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
