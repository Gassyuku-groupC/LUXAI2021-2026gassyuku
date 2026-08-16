#!/usr/bin/env python3
"""Dry-run a conservative safety gate from candidate-action suggestions.

This script only writes what the gate would have done. It does not edit replays
or agent code. It is intentionally narrow: intervene only when the state is
high-risk and the actual action increases scale or upkeep.

The current candidate scorer is mostly a state-risk scorer, not yet a reliable
action-delta scorer. Delta columns are still recorded for analysis, but by
default the gate does not require no_expand to have a lower predicted risk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "rank",
    "turn",
    "turns_to_night",
    "city_tiles",
    "workers",
    "research",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "low_fuel_city_lt5",
    "future_team_loss_20",
    "final_city_tile_margin",
    "actual_big_risk",
    "best_big_risk",
    "actual_error_risk",
    "best_error_risk",
    "actual_safe_expansion",
    "best_expand_safe_expansion",
    "no_expand_big_risk",
    "no_expand_error_risk",
    "no_expand_success",
    "bw_big_risk",
    "bcity_big_risk",
    "research_big_risk",
]


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    for column in ["file", "source_opponent", "phase", "actual_action", "best_risk_action"]:
        if column in data:
            data[column] = data[column].fillna("").astype(str)
    return data


def decide(row: pd.Series, args: argparse.Namespace) -> tuple[str, str, str]:
    actual = str(row.get("actual_action", ""))
    if actual not in {"bw", "bcity", "research"}:
        return "keep", "", "actual action is not gated"

    late = float(row["turn"]) >= args.late_turn
    endgame = float(row["turn"]) >= args.endgame_turn
    low_buffer = float(row["p25_city_fuel_turns"]) < args.low_p25_fuel or float(row["min_city_fuel_turns"]) < args.low_min_fuel
    high_risk = (
        float(row["actual_big_risk"]) >= args.high_big_risk
        or float(row["actual_error_risk"]) >= args.high_error_risk
    )
    no_expand_better = (
        float(row["actual_big_risk"] - row["no_expand_big_risk"]) >= args.min_big_risk_delta
        or float(row["actual_error_risk"] - row["no_expand_error_risk"]) >= args.min_error_risk_delta
    )
    if args.require_risk_delta and not no_expand_better:
        return "keep", "", "risk high but scorer has no action-delta support"
    if late and low_buffer and high_risk:
        return "would_gate", "no_expand", "late low-buffer risky action"
    if endgame and high_risk:
        return "would_gate", "no_expand", "endgame preserve lead/survival mode"
    return "keep", "", "risk below dry-run gate threshold"


def summarize(decisions: pd.DataFrame) -> dict:
    gated = decisions[decisions["gate_decision"] == "would_gate"]
    summary = {
        "states": int(len(decisions)),
        "would_gate": int(len(gated)),
        "gate_rate": float(len(gated) / max(len(decisions), 1)),
        "by_decision": decisions["gate_decision"].value_counts().to_dict(),
        "by_reason": decisions["gate_reason"].value_counts().to_dict(),
        "by_actual_action": decisions.groupby("actual_action")["gate_decision"].value_counts().unstack(fill_value=0).to_dict("index"),
        "by_source_opponent": decisions.groupby("source_opponent")["gate_decision"].value_counts().unstack(fill_value=0).to_dict("index"),
    }
    if not gated.empty:
        summary["gated_outcome_proxy"] = {
            "mean_future_loss20": float(gated["future_team_loss_20"].mean()),
            "loss20_rate": float((gated["future_team_loss_20"] > 0).mean()),
            "loss_rank_rate": float((gated["rank"] == 2).mean()),
            "mean_final_margin": float(gated["final_city_tile_margin"].mean()),
            "mean_actual_big_risk": float(gated["actual_big_risk"].mean()),
            "mean_no_expand_big_risk": float(gated["no_expand_big_risk"].mean()),
            "mean_actual_error_risk": float(gated["actual_error_risk"].mean()),
            "mean_no_expand_error_risk": float(gated["no_expand_error_risk"].mean()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run conservative action gate v1.")
    parser.add_argument("--input", type=Path, default=Path("outputs/diagnostic_layer/best_agent_candidate_action_suggestions_v2_16/candidate_action_suggestions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/dry_run_gate_v1"))
    parser.add_argument("--late-turn", type=int, default=160)
    parser.add_argument("--endgame-turn", type=int, default=280)
    parser.add_argument("--low-p25-fuel", type=float, default=12.0)
    parser.add_argument("--low-min-fuel", type=float, default=3.0)
    parser.add_argument("--high-big-risk", type=float, default=0.35)
    parser.add_argument("--high-error-risk", type=float, default=0.20)
    parser.add_argument("--min-big-risk-delta", type=float, default=0.08)
    parser.add_argument("--min-error-risk-delta", type=float, default=0.04)
    parser.add_argument("--require-risk-delta", action="store_true")
    args = parser.parse_args()

    data = load_data(args.input)
    rows = []
    for _, row in data.iterrows():
        decision, replacement, reason = decide(row, args)
        out = row.to_dict()
        out["big_risk_delta_vs_no_expand"] = (
            float(row.get("actual_big_risk", 0.0)) - float(row.get("no_expand_big_risk", 0.0))
        )
        out["error_risk_delta_vs_no_expand"] = (
            float(row.get("actual_error_risk", 0.0)) - float(row.get("no_expand_error_risk", 0.0))
        )
        out["gate_decision"] = decision
        out["replacement_action"] = replacement
        out["gate_reason"] = reason
        rows.append(out)
    decisions = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(args.output_dir / "dry_run_gate_decisions.csv", index=False, encoding="utf-8")
    gated = decisions[decisions["gate_decision"] == "would_gate"].copy()
    gated.to_csv(args.output_dir / "dry_run_gate_would_intervene.csv", index=False, encoding="utf-8")
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "params": vars(args) | {"input": str(args.input), "output_dir": str(args.output_dir)},
        "summary": summarize(decisions),
    }
    (args.output_dir / "dry_run_gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
