#!/usr/bin/env python3
"""Build conservative intervention/alternative labels from candidate scores.

The output answers two separate questions for each gateable state:
1. Should the frozen actor be overridden?
2. Which compatible macro alternative should receive the removed probability?

These are diagnostic/offline labels. Runtime RL remains responsible for
learning the spatial intervention gate from actual match outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GATE_ALTERNATIVES = {
    "bw": ("research", "no_expand"),
    "bcity": ("no_expand",),
}
PRIMITIVE_ALTERNATIVES = {
    ("bw", "research"): "RESEARCH",
    ("bw", "no_expand"): "NO-OP",
    ("bcity", "no_expand"): "BEST_NON_BUILD_CITY",
}


def numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def prepare_scores(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = data.copy()
    required = {"state_id", "candidate_action", "actual_action"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate score CSV is missing columns: {missing}")
    for column in (
        "p_risk_big_loss_20",
        "p_error_failed_big_loss",
        "p_risk_city_loss_20",
        "p_success_stable_scale",
        "p_safe_expansion_success_40",
    ):
        frame[column] = numeric(frame, column)
    frame["counterfactual_bad_score"] = (
        args.big_loss_weight * frame["p_risk_big_loss_20"]
        + args.error_weight * frame["p_error_failed_big_loss"]
        + args.city_loss_weight * frame["p_risk_city_loss_20"]
        - args.success_credit * frame["p_success_stable_scale"]
        - args.safe_expansion_credit * frame["p_safe_expansion_success_40"]
    )
    return frame


def first_value(group: pd.DataFrame, name: str, default=0):
    if name not in group:
        return default
    value = group.iloc[0][name]
    return default if pd.isna(value) else value


def build_one(group: pd.DataFrame, args: argparse.Namespace) -> dict | None:
    actual_action = str(first_value(group, "actual_action", ""))
    alternatives = GATE_ALTERNATIVES.get(actual_action)
    if alternatives is None:
        return None

    actual_rows = group[group["candidate_action"].astype(str) == actual_action]
    alternative_rows = group[group["candidate_action"].astype(str).isin(alternatives)]
    if actual_rows.empty or alternative_rows.empty:
        return None
    actual = actual_rows.sort_values("counterfactual_bad_score").iloc[0]
    alternative = alternative_rows.sort_values(
        ["counterfactual_bad_score", "p_risk_big_loss_20", "p_error_failed_big_loss"]
    ).iloc[0]

    bad_delta = float(actual["counterfactual_bad_score"] - alternative["counterfactual_bad_score"])
    risk_delta = float(actual["p_risk_big_loss_20"] - alternative["p_risk_big_loss_20"])
    error_delta = float(actual["p_error_failed_big_loss"] - alternative["p_error_failed_big_loss"])
    rank = int(float(first_value(group, "rank", 0)))
    final_margin = float(first_value(group, "final_city_tile_margin", 0.0))
    future_loss20 = float(first_value(group, "future_team_loss_20", 0.0))
    observed_big_loss = bool(
        float(first_value(group, "risk_big_loss_20", 0.0)) > 0
        or float(first_value(group, "error_failed_with_big_loss", 0.0)) > 0
        or future_loss20 >= args.observed_loss_threshold
    )
    losing_or_weak = rank == 2 or final_margin < args.weak_final_margin
    actual_high_risk = (
        float(actual["p_risk_big_loss_20"]) >= args.high_big_risk
        or float(actual["p_error_failed_big_loss"]) >= args.high_error_risk
    )
    material_improvement = (
        bad_delta >= args.min_bad_score_delta
        and (risk_delta >= args.min_risk_delta or error_delta >= args.min_error_delta)
    )
    safe_actual = (
        rank == 1
        and final_margin >= args.safe_final_margin
        and not observed_big_loss
    )

    # Replay outcome is the primary supervision. Candidate-score deltas are
    # useful confidence signals, but the current coarse scorers often assign
    # nearly identical probabilities to bw/bcity/no_expand in the same state.
    # Requiring a delta would therefore erase the exact failed-loss examples
    # the gate is intended to learn from.
    positive = observed_big_loss and losing_or_weak
    negative = safe_actual
    trainable = positive or negative
    if positive:
        reason = (
            "intervene_harmful_actual_with_scored_alternative"
            if material_improvement or actual_high_risk
            else "intervene_harmful_actual_replay_outcome"
        )
    elif safe_actual:
        reason = "keep_actual_safe_winning_outcome"
    else:
        reason = "ambiguous_outcome"

    weight = 1.0
    if positive:
        weight += min(max(bad_delta, 0.0) * args.delta_weight_scale, args.max_delta_weight)
        if material_improvement or actual_high_risk:
            weight += args.scorer_support_bonus
        if rank == 2:
            weight += args.loss_weight_bonus
        if future_loss20 >= args.observed_loss_threshold:
            weight += args.big_loss_weight_bonus

    alternative_action = str(alternative["candidate_action"])
    return {
        "state_id": int(float(first_value(group, "state_id", 0))),
        "file": str(first_value(group, "file", "")),
        "episode_id": str(first_value(group, "episode_id", "")),
        "map_size": int(float(first_value(group, "map_size", 0))),
        "turn": int(float(first_value(group, "turn", 0))),
        "phase": str(first_value(group, "phase", "")),
        "eval_side": str(first_value(group, "eval_side", "")),
        "team": first_value(group, "team", ""),
        "rank": rank,
        "final_city_tile_margin": final_margin,
        "future_team_loss_20": future_loss20,
        "actual_action": actual_action,
        "alternative_action": alternative_action,
        "alternative_primitive": PRIMITIVE_ALTERNATIVES[(actual_action, alternative_action)],
        "intervene_label": int(positive),
        "trainable_label": int(trainable),
        "sample_weight": round(min(weight, args.max_weight), 4),
        "label_reason": reason,
        "actual_bad_score": float(actual["counterfactual_bad_score"]),
        "alternative_bad_score": float(alternative["counterfactual_bad_score"]),
        "bad_score_delta": bad_delta,
        "actual_big_risk": float(actual["p_risk_big_loss_20"]),
        "alternative_big_risk": float(alternative["p_risk_big_loss_20"]),
        "risk_delta": risk_delta,
        "actual_error_risk": float(actual["p_error_failed_big_loss"]),
        "alternative_error_risk": float(alternative["p_error_failed_big_loss"]),
        "error_delta": error_delta,
        "observed_big_loss": int(observed_big_loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learned-gate counterfactual labels.")
    parser.add_argument("--input", type=Path, required=True, help="candidate_action_scores.csv")
    parser.add_argument(
        "--output-csv", type=Path,
        default=Path("dataset/processed/action_gate_counterfactual_labels_v1.csv"),
    )
    parser.add_argument(
        "--summary-json", type=Path,
        default=Path("dataset/processed/action_gate_counterfactual_labels_v1_summary.json"),
    )
    parser.add_argument("--big-loss-weight", type=float, default=1.0)
    parser.add_argument("--error-weight", type=float, default=0.8)
    parser.add_argument("--city-loss-weight", type=float, default=0.25)
    parser.add_argument("--success-credit", type=float, default=0.15)
    parser.add_argument("--safe-expansion-credit", type=float, default=0.10)
    parser.add_argument("--high-big-risk", type=float, default=0.25)
    parser.add_argument("--high-error-risk", type=float, default=0.15)
    parser.add_argument("--min-bad-score-delta", type=float, default=0.08)
    parser.add_argument("--min-risk-delta", type=float, default=0.05)
    parser.add_argument("--min-error-delta", type=float, default=0.03)
    parser.add_argument("--observed-loss-threshold", type=float, default=4.0)
    parser.add_argument("--weak-final-margin", type=float, default=0.0)
    parser.add_argument("--safe-final-margin", type=float, default=5.0)
    parser.add_argument("--delta-weight-scale", type=float, default=2.0)
    parser.add_argument("--max-delta-weight", type=float, default=2.0)
    parser.add_argument("--loss-weight-bonus", type=float, default=1.0)
    parser.add_argument("--big-loss-weight-bonus", type=float, default=1.0)
    parser.add_argument("--scorer-support-bonus", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=5.0)
    args = parser.parse_args()

    scores = prepare_scores(pd.read_csv(args.input, low_memory=False), args)
    rows = []
    for _, group in scores.groupby("state_id", sort=False):
        row = build_one(group, args)
        if row is not None:
            rows.append(row)
    labels = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_csv, index=False, encoding="utf-8")

    trainable = labels[labels["trainable_label"] > 0] if not labels.empty else labels
    summary = {
        "input": str(args.input),
        "output": str(args.output_csv),
        "states": int(scores["state_id"].nunique()),
        "gateable_states": int(len(labels)),
        "trainable_states": int(len(trainable)),
        "positive_states": int(trainable["intervene_label"].sum()) if not trainable.empty else 0,
        "positive_rate": float(trainable["intervene_label"].mean()) if not trainable.empty else 0.0,
        "actual_actions": labels["actual_action"].value_counts().to_dict() if not labels.empty else {},
        "alternatives": labels["alternative_action"].value_counts().to_dict() if not labels.empty else {},
        "reasons": labels["label_reason"].value_counts().to_dict() if not labels.empty else {},
        "by_map_size": (
            trainable.groupby("map_size")["intervene_label"]
            .agg(["count", "sum", "mean"]).to_dict("index")
            if not trainable.empty else {}
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
