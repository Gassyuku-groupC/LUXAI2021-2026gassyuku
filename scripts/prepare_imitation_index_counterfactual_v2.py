#!/usr/bin/env python3
"""Counterfactual v2 imitation reweighting.

V2 keeps the useful fuel-support risk signal from v1, but avoids further
discouraging worker production. It focuses penalties on late macro-risk actions
that can expand upkeep or delay fuel stabilization, and adds small positive
weight to safe high-risk-window examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_imitation_index_counterfactual_v1 import (  # noqa: E402
    action_flags,
    as_float,
    as_int,
    exact_label_key,
    index_key,
    load_counterfactual_labels,
    phase_key,
    replay_actions,
    risk_window,
    turn_bucket,
)


def city_growth(row: dict) -> int:
    return max(as_int(row, "city_tiles_after") - as_int(row, "city_tiles_before"), 0)


def build_label_profiles(labels: list[dict]) -> tuple[dict, dict]:
    exact: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    accum: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {
            "rows": 0,
            "penalty": 0,
            "neutral": 0,
            "safe_late": 0,
            "support_neutral": 0,
            "weight_sum": 0.0,
        }
    )
    for row in labels:
        exact[exact_label_key(row)].append(row)
        key = phase_key(as_int(row, "team"), as_int(row, "turn"), str(row.get("label_source", "")))
        item = accum[key]
        item["rows"] += 1
        if as_int(row, "penalty_label"):
            item["penalty"] += 1
            item["weight_sum"] += as_float(row, "penalty_weight")
        else:
            item["neutral"] += 1
            label = str(row.get("counterfactual_label", ""))
            if label == "late_risk_without_big_loss":
                item["safe_late"] += 1
            elif label == "ignored_support_without_big_loss":
                item["support_neutral"] += 1

    profile = {}
    for key, item in accum.items():
        rows = max(int(item["rows"]), 1)
        penalty = int(item["penalty"])
        profile[key] = {
            **item,
            "penalty_rate": penalty / rows,
            "safe_late_rate": int(item["safe_late"]) / rows,
            "support_neutral_rate": int(item["support_neutral"]) / rows,
            "mean_penalty_weight": item["weight_sum"] / max(penalty, 1),
        }
    return exact, profile


def exact_scale(labels: list[dict], flags: dict[str, int], args: argparse.Namespace) -> tuple[float, list[str], int, int]:
    penalties = [row for row in labels if as_int(row, "penalty_label")]
    safe_late = [row for row in labels if str(row.get("counterfactual_label", "")) == "late_risk_without_big_loss"]
    scale = 1.0
    reasons = []
    if penalties:
        max_weight = max(as_float(row, "penalty_weight") for row in penalties)
        # Exact matches are rare, but still avoid punishing bw-only examples.
        if flags["has_bcity"] or flags["has_research"] or not flags["has_bw"]:
            factor = max(args.exact_min_scale, 1.0 - args.exact_penalty_strength * min(max_weight / 3.0, 1.0))
            scale *= factor
            reasons.append(f"counterfactual_v2_exact_penalty_x{factor:.3f}")
    if safe_late and not penalties:
        factor = min(args.exact_safe_max_scale, 1.0 + args.exact_safe_strength)
        scale *= factor
        reasons.append(f"counterfactual_v2_exact_safe_x{factor:.3f}")
    if not reasons:
        reasons.append("counterfactual_v2_exact_neutral")
    return scale, reasons, len(penalties), len(safe_late)


def phase_scale(row: dict, flags: dict[str, int], profile: dict, args: argparse.Namespace) -> tuple[float, list[str], int, int]:
    turn = as_int(row, "state_step")
    team = as_int(row, "teacher_player")
    max_teacher_night_loss = as_int(row, "max_teacher_night_loss")
    night_loss_next = as_int(row, "night_city_loss_next")
    growth = city_growth(row)
    safe_teacher = max_teacher_night_loss <= args.safe_teacher_max_night_loss and night_loss_next <= 0
    safe_boost_teacher = max_teacher_night_loss <= args.safe_boost_max_night_loss and night_loss_next <= 0

    scale = 1.0
    reasons = []
    penalty_hits = 0
    safe_hits = 0

    late_item = profile.get(phase_key(team, turn, "late_big_loss_warning"))
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
            if risky_late_action or inefficient_growth or night_loss_next > 0:
                penalty_hits += 1
                factor = max(args.phase_min_scale, 1.0 - args.phase_late_strength * phase_strength)
                scale *= factor
                reasons.append(f"counterfactual_v2_late_penalty_x{factor:.3f}")

        if has_late_safe_profile and safe_boost_teacher and turn >= args.start_turn:
            safe_development = growth > 0 or flags["has_transfer"]
            if safe_development:
                safe_hits += 1
                factor = min(args.safe_phase_max_scale, 1.0 + args.safe_late_strength)
                if growth > 0:
                    factor = min(args.safe_phase_max_scale, factor + args.safe_growth_extra)
                scale *= factor
                reasons.append(f"counterfactual_v2_late_safe_x{factor:.3f}")

    support_item = profile.get(phase_key(team, turn, "suggest_fuel_support"))
    if support_item:
        has_support_penalty_profile = (
            int(support_item["penalty"]) >= args.min_phase_penalties
            and float(support_item["penalty_rate"]) >= args.min_phase_penalty_rate
        )
        if has_support_penalty_profile:
            phase_strength = min(float(support_item["mean_penalty_weight"]) / 3.0, 1.0)
            unsupported_window = (
                risk_window(turn, args.risk_window_lead_turns)
                and not flags["has_transfer"]
                and (flags["has_bcity"] or flags["has_research"])
            )
            if unsupported_window or (night_loss_next > 0 and not flags["has_transfer"] and not flags["has_bw"]):
                penalty_hits += 1
                factor = max(args.support_phase_min_scale, 1.0 - args.phase_support_strength * phase_strength)
                scale *= factor
                reasons.append(f"counterfactual_v2_support_penalty_x{factor:.3f}")

        # Small positive signal for support-like behavior in risky phases that
        # did not collapse. This keeps v2 from becoming simply "do less".
        if (
            int(support_item["support_neutral"]) >= args.min_safe_phase_rows
            and safe_boost_teacher
            and risk_window(turn, args.risk_window_lead_turns)
            and flags["has_transfer"]
        ):
            safe_hits += 1
            factor = min(args.safe_phase_max_scale, 1.0 + args.safe_support_strength)
            scale *= factor
            reasons.append(f"counterfactual_v2_support_safe_x{factor:.3f}")

    if not reasons:
        reasons.append("counterfactual_v2_no_action_match")
    return scale, reasons, penalty_hits, safe_hits


def process_index(args: argparse.Namespace) -> None:
    labels = load_counterfactual_labels(args.labels)
    exact, profile = build_label_profiles(labels)
    replay_cache: dict[Path, dict] = {}
    rows = []
    stats = Counter()
    phase_keys_hit = Counter()
    safe_keys_hit = Counter()

    with args.input.open(encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if args.map_size and as_int(row, "width") != args.map_size:
                rows.append(row)
                continue

            original_weight = as_float(row, "weight", 1.0)
            weight = original_weight
            reasons = [row.get("weight_reason", "")]
            flags = {"has_bw": 0, "has_bcity": 0, "has_research": 0, "has_transfer": 0, "has_move": 0}
            exact_labels = exact.get(index_key(row), [])
            exact_penalties = 0
            exact_safe = 0
            phase_penalty_hits = 0
            phase_safe_hits = 0

            actions = replay_actions(row, replay_cache)
            flags = action_flags(actions)
            if exact_labels:
                scale, exact_reasons, exact_penalties, exact_safe = exact_scale(exact_labels, flags, args)
                weight *= scale
                reasons.extend(exact_reasons)
                stats["exact_matched_rows"] += 1
                if exact_penalties:
                    stats["exact_penalty_rows"] += 1
                if exact_safe:
                    stats["exact_safe_rows"] += 1
            elif args.enable_phase_fallback:
                scale, phase_reasons, phase_penalty_hits, phase_safe_hits = phase_scale(row, flags, profile, args)
                weight *= scale
                reasons.extend(phase_reasons)
                if phase_penalty_hits and scale < 1.0:
                    stats["phase_penalty_reweighted_rows"] += 1
                    phase_keys_hit[(as_int(row, "teacher_player"), turn_bucket(as_int(row, "state_step")))] += 1
                if phase_safe_hits and scale > 1.0:
                    stats["phase_safe_reweighted_rows"] += 1
                    safe_keys_hit[(as_int(row, "teacher_player"), turn_bucket(as_int(row, "state_step")))] += 1
            else:
                reasons.append("counterfactual_v2_no_exact_match")

            if weight != original_weight:
                stats["changed_rows"] += 1
            stats["rows"] += 1
            out = dict(row)
            out["original_weight"] = f"{original_weight:.4f}"
            out["counterfactual_exact_labels"] = str(len(exact_labels))
            out["counterfactual_exact_penalties"] = str(exact_penalties)
            out["counterfactual_exact_safe"] = str(exact_safe)
            out["counterfactual_phase_penalty_hits"] = str(phase_penalty_hits)
            out["counterfactual_phase_safe_hits"] = str(phase_safe_hits)
            for key, value in flags.items():
                out[key] = str(value)
            out["counterfactual_scale"] = f"{(weight / original_weight if original_weight else 0.0):.4f}"
            out["weight"] = f"{min(max(weight, args.min_weight), args.max_weight):.4f}"
            out["weight_reason"] = "+".join(part for part in reasons if part)
            rows.append(out)

    extras = [
        "original_weight",
        "counterfactual_exact_labels",
        "counterfactual_exact_penalties",
        "counterfactual_exact_safe",
        "counterfactual_phase_penalty_hits",
        "counterfactual_phase_safe_hits",
        "has_bw",
        "has_bcity",
        "has_research",
        "has_transfer",
        "has_move",
        "counterfactual_scale",
    ]
    out_fields = [field for field in fieldnames if field not in extras] + extras
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "version": "counterfactual_v2",
        "input": str(args.input),
        "output": str(args.output),
        "labels": str(args.labels),
        "label_rows": len(labels),
        "exact_label_keys": len(exact),
        "phase_profile_keys": len(profile),
        "stats": dict(stats),
        "phase_penalty_by_side_bucket": {f"p{team}:{bucket}": count for (team, bucket), count in phase_keys_hit.items()},
        "phase_safe_by_side_bucket": {f"p{team}:{bucket}": count for (team, bucket), count in safe_keys_hit.items()},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply counterfactual v2 risk/safe-development reweighting.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_hq.csv"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_risk_labels_v1_from_best/counterfactual_risk_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_counterfactual_v2.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_reweight_v2/summary.json"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--start-turn", type=int, default=140)
    parser.add_argument("--enable-phase-fallback", action="store_true", default=True)
    parser.add_argument("--min-phase-penalties", type=int, default=2)
    parser.add_argument("--min-phase-penalty-rate", type=float, default=0.50)
    parser.add_argument("--min-safe-phase-rows", type=int, default=1)
    parser.add_argument("--safe-teacher-max-night-loss", type=int, default=10)
    parser.add_argument("--safe-boost-max-night-loss", type=int, default=5)
    parser.add_argument("--risk-window-lead-turns", type=int, default=5)
    parser.add_argument("--exact-penalty-strength", type=float, default=0.40)
    parser.add_argument("--exact-min-scale", type=float, default=0.55)
    parser.add_argument("--exact-safe-strength", type=float, default=0.08)
    parser.add_argument("--exact-safe-max-scale", type=float, default=1.12)
    parser.add_argument("--phase-late-strength", type=float, default=0.16)
    parser.add_argument("--phase-support-strength", type=float, default=0.10)
    parser.add_argument("--phase-min-scale", type=float, default=0.78)
    parser.add_argument("--support-phase-min-scale", type=float, default=0.84)
    parser.add_argument("--safe-late-strength", type=float, default=0.06)
    parser.add_argument("--safe-growth-extra", type=float, default=0.03)
    parser.add_argument("--safe-support-strength", type=float, default=0.05)
    parser.add_argument("--safe-phase-max-scale", type=float, default=1.12)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=5.0)
    args = parser.parse_args()
    process_index(args)


if __name__ == "__main__":
    main()
