#!/usr/bin/env python3
"""Counterfactual v4 imitation reweighting.

V4 blends v2's safety with v3's worker recovery. It keeps bw-only rows from
being penalized, but reduces v3's broad safe boost. Late macro penalties are
aimed narrowly at bcity/research/inefficient growth, with extra caution when a
teacher trajectory soon loses city tiles.
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
        # Keep worker production plasticity; exact penalties only touch macro
        # choices that can expand upkeep or delay fuel stabilization.
        if flags["has_bcity"] or flags["has_research"]:
            factor = max(args.exact_min_scale, 1.0 - args.exact_penalty_strength * min(max_weight / 3.0, 1.0))
            scale *= factor
            reasons.append(f"counterfactual_v4_exact_penalty_x{factor:.3f}")

    if safe_late and not penalties:
        if flags["has_transfer"] or flags["has_bw"]:
            factor = min(args.exact_safe_max_scale, 1.0 + args.exact_safe_strength)
            scale *= factor
            reasons.append(f"counterfactual_v4_exact_safe_x{factor:.3f}")

    if not reasons:
        reasons.append("counterfactual_v4_exact_neutral")
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
            imminent_loss = night_loss_next > 0 and (flags["has_bcity"] or flags["has_research"] or growth > 0)
            if risky_late_action or inefficient_growth or imminent_loss:
                penalty_hits += 1
                strength = args.phase_late_strength
                if night_loss_next > args.big_loss_next_threshold:
                    strength += args.imminent_loss_extra_strength
                factor = max(args.phase_min_scale, 1.0 - strength * phase_strength)
                scale *= factor
                reasons.append(f"counterfactual_v4_late_penalty_x{factor:.3f}")

        if has_late_safe_profile and safe_boost_teacher and turn >= args.start_turn:
            # V4 safe boost is intentionally narrower than v3: support and
            # bw-only get help, broad growth does not automatically get it.
            bw_only = flags["has_bw"] and not flags["has_bcity"] and not flags["has_research"]
            support_like = flags["has_transfer"]
            if bw_only or support_like:
                safe_hits += 1
                factor = 1.0 + args.safe_late_strength
                if bw_only:
                    factor += args.safe_bw_extra
                if support_like:
                    factor += args.safe_support_extra
                factor = min(args.safe_phase_max_scale, factor)
                scale *= factor
                reasons.append(f"counterfactual_v4_late_safe_x{factor:.3f}")

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
                reasons.append(f"counterfactual_v4_support_penalty_x{factor:.3f}")

        if (
            int(support_item["support_neutral"]) >= args.min_safe_phase_rows
            and safe_teacher
            and v2.risk_window(turn, args.risk_window_lead_turns)
            and flags["has_transfer"]
        ):
            safe_hits += 1
            factor = min(args.safe_phase_max_scale, 1.0 + args.safe_support_strength)
            scale *= factor
            reasons.append(f"counterfactual_v4_support_safe_x{factor:.3f}")

    if not reasons:
        reasons.append("counterfactual_v4_no_action_match")
    return scale, reasons, penalty_hits, safe_hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply counterfactual v4 balanced risk/scale reweighting.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_hq.csv"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_risk_labels_v1_from_best/counterfactual_risk_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_counterfactual_v4.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_reweight_v4/summary.json"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--start-turn", type=int, default=140)
    parser.add_argument("--enable-phase-fallback", action="store_true", default=True)
    parser.add_argument("--min-phase-penalties", type=int, default=2)
    parser.add_argument("--min-phase-penalty-rate", type=float, default=0.58)
    parser.add_argument("--min-safe-phase-rows", type=int, default=1)
    parser.add_argument("--safe-teacher-max-night-loss", type=int, default=10)
    parser.add_argument("--safe-boost-max-night-loss", type=int, default=6)
    parser.add_argument("--risk-window-lead-turns", type=int, default=5)
    parser.add_argument("--big-loss-next-threshold", type=int, default=10)
    parser.add_argument("--exact-penalty-strength", type=float, default=0.32)
    parser.add_argument("--exact-min-scale", type=float, default=0.66)
    parser.add_argument("--exact-safe-strength", type=float, default=0.07)
    parser.add_argument("--exact-safe-max-scale", type=float, default=1.12)
    parser.add_argument("--phase-late-strength", type=float, default=0.13)
    parser.add_argument("--imminent-loss-extra-strength", type=float, default=0.04)
    parser.add_argument("--phase-support-strength", type=float, default=0.08)
    parser.add_argument("--phase-min-scale", type=float, default=0.82)
    parser.add_argument("--support-phase-min-scale", type=float, default=0.90)
    parser.add_argument("--safe-late-strength", type=float, default=0.035)
    parser.add_argument("--safe-bw-extra", type=float, default=0.025)
    parser.add_argument("--safe-support-extra", type=float, default=0.015)
    parser.add_argument("--safe-support-strength", type=float, default=0.04)
    parser.add_argument("--safe-phase-max-scale", type=float, default=1.10)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=5.0)
    args = parser.parse_args()

    v2.exact_scale = exact_scale
    v2.phase_scale = phase_scale
    v2.process_index(args)
    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        summary["version"] = "counterfactual_v4"
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
