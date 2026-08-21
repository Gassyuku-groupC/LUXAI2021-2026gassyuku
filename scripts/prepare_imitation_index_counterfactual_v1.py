#!/usr/bin/env python3
"""Reweight imitation index rows using conservative counterfactual risk labels.

The label source may come from best-agent diagnostic replays while the imitation
index may come from public Kaggle replay data. Therefore this script first tries
exact episode/team/turn matching, then falls back to a conservative phase-level
profile only when exact matching is unavailable.
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

from replay_dataset_utils import load_replay  # noqa: E402


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    return int(as_float(row, key, float(default)))


def turn_bucket(turn: int) -> str:
    if turn < 40:
        return "000-039"
    if turn < 80:
        return "040-079"
    if turn < 120:
        return "080-119"
    if turn < 140:
        return "120-139"
    if turn < 160:
        return "140-159"
    if turn < 200:
        return "160-199"
    if turn < 240:
        return "200-239"
    if turn < 280:
        return "240-279"
    if turn < 320:
        return "280-319"
    return "320-360"


def risk_window(turn: int, lead_turns: int) -> bool:
    return turn % 40 >= 30 - lead_turns


def load_counterfactual_labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def exact_label_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("episode_id", "")),
        str(as_int(row, "team")),
        str(as_int(row, "turn")),
    )


def index_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("episode_id", "")),
        str(as_int(row, "teacher_player")),
        str(as_int(row, "state_step")),
    )


def phase_key(team: int, turn: int, label_source: str) -> tuple[str, str, str]:
    return (str(team), turn_bucket(turn), label_source)


def build_label_profiles(labels: list[dict]) -> tuple[dict, dict]:
    exact: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    profile: dict[tuple[str, str, str], dict] = {}
    accum: dict[tuple[str, str, str], dict] = defaultdict(lambda: {"rows": 0, "penalty": 0, "neutral": 0, "weight_sum": 0.0})
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
    for key, item in accum.items():
        rows = max(int(item["rows"]), 1)
        penalty = int(item["penalty"])
        profile[key] = {
            **item,
            "penalty_rate": penalty / rows,
            "mean_penalty_weight": item["weight_sum"] / max(penalty, 1),
        }
    return exact, profile


def replay_actions(row: dict, cache: dict[Path, dict]) -> list[str]:
    path = Path(row["file"])
    if path not in cache:
        cache[path] = load_replay(path)
    replay = cache[path]
    action_step = as_int(row, "action_step")
    player = as_int(row, "teacher_player")
    try:
        return [str(action) for action in replay["steps"][action_step][player].get("action") or []]
    except (KeyError, IndexError, TypeError):
        return []


def action_flags(actions: list[str]) -> dict[str, int]:
    flags = {
        "has_bw": 0,
        "has_bcity": 0,
        "has_research": 0,
        "has_transfer": 0,
        "has_move": 0,
    }
    for action in actions:
        parts = action.split()
        if not parts:
            continue
        kind = parts[0]
        if kind == "bw":
            flags["has_bw"] = 1
        elif kind == "bcity":
            flags["has_bcity"] = 1
        elif kind in {"r", "bc"}:
            flags["has_research"] = 1
        elif kind == "t":
            flags["has_transfer"] = 1
        elif kind == "m":
            flags["has_move"] = 1
    return flags


def exact_scale(labels: list[dict], args: argparse.Namespace) -> tuple[float, str, int]:
    penalties = [row for row in labels if as_int(row, "penalty_label")]
    if not penalties:
        return 1.0, "counterfactual_exact_neutral", 0
    max_weight = max(as_float(row, "penalty_weight") for row in penalties)
    scale = max(args.exact_min_scale, 1.0 - args.exact_penalty_strength * min(max_weight / 3.0, 1.0))
    return scale, "counterfactual_exact_penalty", len(penalties)


def phase_scale(row: dict, flags: dict[str, int], profile: dict, args: argparse.Namespace) -> tuple[float, list[str], int]:
    turn = as_int(row, "state_step")
    team = as_int(row, "teacher_player")
    max_teacher_night_loss = as_int(row, "max_teacher_night_loss")
    night_loss_next = as_int(row, "night_city_loss_next")
    if max_teacher_night_loss <= args.min_teacher_max_night_loss and night_loss_next <= 0:
        return 1.0, ["counterfactual_phase_safe_teacher_skip"], 0

    scale = 1.0
    reasons = []
    hits = 0
    for label_source in ("late_big_loss_warning", "suggest_fuel_support"):
        item = profile.get(phase_key(team, turn, label_source))
        if not item:
            continue
        if int(item["penalty"]) < args.min_phase_penalties:
            continue
        if float(item["penalty_rate"]) < args.min_phase_penalty_rate:
            continue
        hits += 1
        phase_strength = min(float(item["mean_penalty_weight"]) / 3.0, 1.0)
        if label_source == "late_big_loss_warning":
            risky_macro_action = flags["has_bw"] or flags["has_bcity"] or (turn >= args.start_turn and flags["has_research"])
            if risky_macro_action or night_loss_next > 0:
                factor = max(args.phase_min_scale, 1.0 - args.phase_late_strength * phase_strength)
                scale *= factor
                reasons.append(f"counterfactual_phase_late_x{factor:.3f}")
        elif label_source == "suggest_fuel_support":
            unsupported_window = (
                risk_window(turn, args.risk_window_lead_turns)
                and not flags["has_transfer"]
                and (flags["has_bw"] or flags["has_bcity"] or flags["has_research"])
            )
            if unsupported_window or night_loss_next > 0:
                factor = max(args.phase_min_scale, 1.0 - args.phase_support_strength * phase_strength)
                scale *= factor
                reasons.append(f"counterfactual_phase_support_x{factor:.3f}")
    if not reasons:
        reasons.append("counterfactual_phase_no_action_match")
    return scale, reasons, hits


def process_index(args: argparse.Namespace) -> None:
    labels = load_counterfactual_labels(args.labels)
    exact, profile = build_label_profiles(labels)
    replay_cache: dict[Path, dict] = {}
    rows = []
    stats = Counter()
    phase_keys_hit = Counter()

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
            exact_labels = exact.get(index_key(row), [])
            exact_count = 0
            phase_count = 0
            flags = {"has_bw": 0, "has_bcity": 0, "has_research": 0, "has_transfer": 0, "has_move": 0}
            if exact_labels:
                scale, reason, exact_count = exact_scale(exact_labels, args)
                weight *= scale
                reasons.append(f"{reason}_x{scale:.3f}")
                stats["exact_matched_rows"] += 1
                if exact_count:
                    stats["exact_penalty_rows"] += 1
            elif args.enable_phase_fallback:
                actions = replay_actions(row, replay_cache)
                flags = action_flags(actions)
                scale, phase_reasons, phase_count = phase_scale(row, flags, profile, args)
                weight *= scale
                reasons.extend(phase_reasons)
                if phase_count and scale < 1.0:
                    stats["phase_reweighted_rows"] += 1
                    phase_keys_hit[(as_int(row, "teacher_player"), turn_bucket(as_int(row, "state_step")))] += 1
            else:
                reasons.append("counterfactual_no_exact_match")

            if weight != original_weight:
                stats["changed_rows"] += 1
            stats["rows"] += 1
            out = dict(row)
            out["original_weight"] = f"{original_weight:.4f}"
            out["counterfactual_exact_labels"] = str(len(exact_labels))
            out["counterfactual_exact_penalties"] = str(exact_count)
            out["counterfactual_phase_hits"] = str(phase_count)
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
        "counterfactual_phase_hits",
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
        "input": str(args.input),
        "output": str(args.output),
        "labels": str(args.labels),
        "label_rows": len(labels),
        "exact_label_keys": len(exact),
        "phase_profile_keys": len(profile),
        "stats": dict(stats),
        "phase_reweighted_by_side_bucket": {f"p{team}:{bucket}": count for (team, bucket), count in phase_keys_hit.items()},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply conservative counterfactual risk reweighting to an imitation index.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_hq.csv"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_risk_labels_v1_from_best/counterfactual_risk_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_counterfactual_v1.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_reweight_v1/summary.json"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--start-turn", type=int, default=140)
    parser.add_argument("--enable-phase-fallback", action="store_true", default=True)
    parser.add_argument("--min-phase-penalties", type=int, default=2)
    parser.add_argument("--min-phase-penalty-rate", type=float, default=0.50)
    parser.add_argument("--min-teacher-max-night-loss", type=int, default=10)
    parser.add_argument("--risk-window-lead-turns", type=int, default=5)
    parser.add_argument("--exact-penalty-strength", type=float, default=0.45)
    parser.add_argument("--exact-min-scale", type=float, default=0.45)
    parser.add_argument("--phase-late-strength", type=float, default=0.18)
    parser.add_argument("--phase-support-strength", type=float, default=0.12)
    parser.add_argument("--phase-min-scale", type=float, default=0.75)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=5.0)
    args = parser.parse_args()
    process_index(args)


if __name__ == "__main__":
    main()
