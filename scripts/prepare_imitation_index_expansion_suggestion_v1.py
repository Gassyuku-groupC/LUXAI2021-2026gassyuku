#!/usr/bin/env python3
"""Apply conservative expansion-suggestion reweighting to an imitation index.

Expansion suggestion labels come from best-agent diagnostic replays, so they do
not exactly overlap the official replay imitation index. This script therefore
uses a phase/turn-bucket profile and only boosts existing teacher rows that
already contain bcity. It never penalizes bw/research/no-expand rows.
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

from prepare_imitation_index_counterfactual_v1 import action_flags, as_float, as_int, replay_actions, turn_bucket  # noqa: E402


def label_turn_bucket(turn: int) -> str:
    return turn_bucket(turn)


def build_profile(labels_path: Path) -> dict[str, dict]:
    accum: dict[str, dict] = defaultdict(
        lambda: {
            "rows": 0,
            "loss_rows": 0,
            "weight_sum": 0.0,
            "safe_sum": 0.0,
            "big_risk_sum": 0.0,
            "p25_sum": 0.0,
            "priority": Counter(),
        }
    )
    with labels_path.open(encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            bucket = label_turn_bucket(as_int(row, "turn"))
            item = accum[bucket]
            weight = as_float(row, "expansion_positive_weight")
            item["rows"] += 1
            item["weight_sum"] += weight
            item["safe_sum"] += as_float(row, "bcity_safe_expansion")
            item["big_risk_sum"] += as_float(row, "bcity_big_risk")
            item["p25_sum"] += as_float(row, "p25_city_fuel_turns")
            item["priority"][str(row.get("priority_bucket", ""))] += 1
            if as_int(row, "rank") == 2:
                item["loss_rows"] += 1

    profile = {}
    for bucket, item in accum.items():
        rows = max(int(item["rows"]), 1)
        profile[bucket] = {
            "rows": int(item["rows"]),
            "loss_rows": int(item["loss_rows"]),
            "loss_rate": item["loss_rows"] / rows,
            "mean_weight": item["weight_sum"] / rows,
            "mean_safe": item["safe_sum"] / rows,
            "mean_big_risk": item["big_risk_sum"] / rows,
            "mean_p25_fuel": item["p25_sum"] / rows,
            "priority_counts": dict(item["priority"]),
        }
    return profile


def expansion_scale(row: dict, flags: dict[str, int], profile: dict[str, dict], args: argparse.Namespace) -> tuple[float, str, int]:
    turn = as_int(row, "state_step")
    bucket = turn_bucket(turn)
    item = profile.get(bucket)
    if not item:
        return 1.0, "expansion_suggestion_no_profile", 0
    if int(item["rows"]) < args.min_profile_rows:
        return 1.0, "expansion_suggestion_profile_too_small", 0
    if not flags["has_bcity"]:
        return 1.0, "expansion_suggestion_non_bcity_unchanged", 0
    if turn < args.min_turn or turn > args.max_turn:
        return 1.0, "expansion_suggestion_outside_turn_range", 0
    if as_int(row, "night_city_loss_next") > args.max_night_loss_next:
        return 1.0, "expansion_suggestion_teacher_not_safe", 0
    if as_int(row, "max_teacher_night_loss") > args.max_teacher_night_loss:
        return 1.0, "expansion_suggestion_teacher_max_loss_high", 0
    if as_int(row, "city_tiles_after") < as_int(row, "city_tiles_before"):
        return 1.0, "expansion_suggestion_no_growth", 0

    base = args.base_strength + float(item["mean_weight"]) * args.profile_weight_strength
    if turn >= args.late_turn:
        base += args.late_extra
    if float(item["loss_rate"]) >= args.loss_profile_rate:
        base += args.loss_profile_extra
    if as_int(row, "rank") == 1:
        base += args.winner_extra
    factor = min(args.max_scale, 1.0 + base)
    return factor, f"expansion_suggestion_v1_{bucket}_x{factor:.3f}", 1


def process(args: argparse.Namespace) -> None:
    profile = build_profile(args.labels)
    replay_cache: dict[Path, dict] = {}
    stats = Counter()
    bucket_hits = Counter()
    rows = []

    with args.input.open(encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if args.map_size and as_int(row, "width") != args.map_size:
                rows.append(row)
                continue

            original_weight = as_float(row, "weight", 1.0)
            actions = replay_actions(row, replay_cache)
            flags = action_flags(actions)
            scale, reason, hit = expansion_scale(row, flags, profile, args)
            weight = min(max(original_weight * scale, args.min_weight), args.max_weight)

            out = dict(row)
            out["expansion_suggestion_original_weight"] = f"{original_weight:.4f}"
            out["expansion_suggestion_scale"] = f"{scale:.4f}"
            out["expansion_suggestion_phase_hit"] = str(hit)
            out["expansion_suggestion_profile_bucket"] = turn_bucket(as_int(row, "state_step"))
            out["has_bw"] = str(flags["has_bw"])
            out["has_bcity"] = str(flags["has_bcity"])
            out["has_research"] = str(flags["has_research"])
            out["has_transfer"] = str(flags["has_transfer"])
            out["has_move"] = str(flags["has_move"])
            out["weight"] = f"{weight:.4f}"
            out["weight_reason"] = "+".join(part for part in [row.get("weight_reason", ""), reason] if part)
            rows.append(out)

            stats["rows"] += 1
            if hit:
                stats["boosted_rows"] += 1
                bucket_hits[turn_bucket(as_int(row, "state_step"))] += 1
            if weight != original_weight:
                stats["changed_rows"] += 1

    extras = [
        "expansion_suggestion_original_weight",
        "expansion_suggestion_scale",
        "expansion_suggestion_phase_hit",
        "expansion_suggestion_profile_bucket",
        "has_bw",
        "has_bcity",
        "has_research",
        "has_transfer",
        "has_move",
    ]
    out_fields = [field for field in fieldnames if field not in extras] + extras
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "version": "expansion_suggestion_v1",
        "input": str(args.input),
        "labels": str(args.labels),
        "output": str(args.output),
        "label_profile": profile,
        "stats": dict(stats),
        "boosted_by_turn_bucket": dict(bucket_hits),
        "params": {
            "min_profile_rows": args.min_profile_rows,
            "min_turn": args.min_turn,
            "max_turn": args.max_turn,
            "late_turn": args.late_turn,
            "max_night_loss_next": args.max_night_loss_next,
            "max_teacher_night_loss": args.max_teacher_night_loss,
            "base_strength": args.base_strength,
            "profile_weight_strength": args.profile_weight_strength,
            "max_scale": args.max_scale,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply conservative expansion-suggestion reweighting.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_counterfactual_v4.csv"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/processed/expansion_suggestion_labels_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_expansion_suggestion_v1.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/diagnostic_layer/expansion_suggestion_reweight_v1/summary.json"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--min-profile-rows", type=int, default=20)
    parser.add_argument("--min-turn", type=int, default=80)
    parser.add_argument("--max-turn", type=int, default=320)
    parser.add_argument("--late-turn", type=int, default=160)
    parser.add_argument("--max-night-loss-next", type=int, default=0)
    parser.add_argument("--max-teacher-night-loss", type=int, default=8)
    parser.add_argument("--base-strength", type=float, default=0.015)
    parser.add_argument("--profile-weight-strength", type=float, default=0.12)
    parser.add_argument("--late-extra", type=float, default=0.010)
    parser.add_argument("--loss-profile-rate", type=float, default=0.20)
    parser.add_argument("--loss-profile-extra", type=float, default=0.008)
    parser.add_argument("--winner-extra", type=float, default=0.004)
    parser.add_argument("--max-scale", type=float, default=1.065)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=5.0)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
