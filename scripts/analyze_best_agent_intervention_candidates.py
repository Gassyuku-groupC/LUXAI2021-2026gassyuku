#!/usr/bin/env python3
"""Extract non-intrusive intervention candidates from diagnostic scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from make_diagnostic_report import load_scores


OUTPUT_COLUMNS = [
    "priority_score",
    "candidate_type",
    "suggestion",
    "reason",
    "file",
    "opponent",
    "seed",
    "eval_side",
    "team",
    "turn",
    "turn_bucket",
    "p_loss_10",
    "p_safe_expansion",
    "future_team_loss_10",
    "city_tiles",
    "workers",
    "worker_citytile_ratio",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "fuel_turns_total",
    "bcity_actions",
    "bw_actions",
    "bw_low_fuel_lt5_actions",
    "bcity_adjacent_low_fuel_lt5_actions",
    "final_city_tiles",
]


def concise_file(path: str) -> str:
    return os.path.basename(str(path))


def add_common_columns(frame: pd.DataFrame, candidate_type: str, suggestion: str) -> pd.DataFrame:
    out = frame.copy()
    out["candidate_type"] = candidate_type
    out["suggestion"] = suggestion
    out["file"] = out["file"].map(concise_file)
    return out


def build_safety_candidates(cand: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mask = (
        cand["p_loss_10"].ge(args.safety_risk_threshold)
        & ((cand["bcity_actions"] > 0) | (cand["bw_actions"] > 0))
    )
    out = add_common_columns(
        cand[mask],
        "safety_gate",
        "log-only first; later consider blocking only bw or low-fuel-adjacent bcity",
    )
    if out.empty:
        return out
    out["priority_score"] = (
        out["p_loss_10"] * 2.0
        + (out["future_team_loss_10"].clip(upper=10) / 10.0)
        + (out["bw_low_fuel_lt5_actions"] > 0).astype(float) * 0.5
        + (out["bcity_adjacent_low_fuel_lt5_actions"] > 0).astype(float) * 0.5
    )
    reasons = []
    for _, row in out.iterrows():
        parts = [f"risk={row['p_loss_10']:.3f}"]
        if row["bw_actions"] > 0:
            parts.append(f"bw={int(row['bw_actions'])}")
        if row["bcity_actions"] > 0:
            parts.append(f"bcity={int(row['bcity_actions'])}")
        if row["bw_low_fuel_lt5_actions"] > 0:
            parts.append("bw_low_fuel_lt5")
        if row["bcity_adjacent_low_fuel_lt5_actions"] > 0:
            parts.append("bcity_adjacent_low_fuel_lt5")
        parts.append(f"future_loss_10={row['future_team_loss_10']:.0f}")
        reasons.append("; ".join(parts))
    out["reason"] = reasons
    return out.sort_values(["priority_score", "p_loss_10"], ascending=[False, False])


def build_expansion_candidates(cand: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mask = (
        cand["p_safe_expansion"].ge(args.safe_expansion_threshold)
        & cand["p_loss_10"].lt(args.low_risk_threshold)
        & cand["bcity_actions"].le(0)
    )
    if args.expansion_min_turn:
        mask &= cand["turn"].ge(args.expansion_min_turn)
    if args.expansion_max_turn:
        mask &= cand["turn"].le(args.expansion_max_turn)
    out = add_common_columns(
        cand[mask],
        "expansion_suggestion",
        "log missed opportunity; do not force bcity until repeated across seeds",
    )
    if out.empty:
        return out
    scale_need = (1.0 - out["worker_citytile_ratio"].clip(lower=0.0, upper=1.0))
    buffer_health = (out["p25_city_fuel_turns"].clip(lower=0.0, upper=20.0) / 20.0)
    out["priority_score"] = out["p_safe_expansion"] * 2.0 + scale_need + buffer_health - out["p_loss_10"]
    reasons = []
    for _, row in out.iterrows():
        parts = [
            f"safe={row['p_safe_expansion']:.3f}",
            f"risk={row['p_loss_10']:.3f}",
            f"ratio={row['worker_citytile_ratio']:.2f}",
            f"p25_fuel={row['p25_city_fuel_turns']:.2f}",
        ]
        if row["worker_citytile_ratio"] < args.low_worker_citytile_ratio:
            parts.append("low_worker_citytile_ratio")
        if row["turn"] >= 240:
            parts.append("late_scale_window")
        reasons.append("; ".join(parts))
    out["reason"] = reasons
    return out.sort_values(["priority_score", "p_safe_expansion"], ascending=[False, False])


def build_watchlist(cand: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mask = cand["p_loss_10"].ge(args.watch_risk_threshold)
    out = add_common_columns(cand[mask], "risk_watch", "log only; inspect fuel routing before any rule")
    if out.empty:
        return out
    out["priority_score"] = out["p_loss_10"] + out["future_team_loss_10"].clip(upper=10) / 10.0
    out["reason"] = out.apply(
        lambda row: (
            f"risk={row['p_loss_10']:.3f}; future_loss_10={row['future_team_loss_10']:.0f}; "
            f"min_fuel={row['min_city_fuel_turns']:.2f}; p25_fuel={row['p25_city_fuel_turns']:.2f}"
        ),
        axis=1,
    )
    return out.sort_values(["priority_score", "p_loss_10"], ascending=[False, False])


def write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        out = table.copy()
        if not out.empty:
            out = out[[column for column in OUTPUT_COLUMNS if column in out.columns]]
        out.head(args.top_n).to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    combined = pd.concat(
        [
            table.head(args.top_n)
            for table in tables.values()
            if not table.empty
        ],
        ignore_index=True,
    )
    if not combined.empty:
        combined = combined[[column for column in OUTPUT_COLUMNS if column in combined.columns]]
        combined = combined.sort_values(["candidate_type", "priority_score"], ascending=[True, False])
    combined.to_csv(output_dir / "intervention_candidates_combined.csv", index=False, encoding="utf-8")

    summary_rows = []
    for name, table in tables.items():
        summary_rows.append({"candidate_type": name, "rows": int(len(table))})
    pd.DataFrame(summary_rows).to_csv(output_dir / "intervention_candidate_summary.csv", index=False, encoding="utf-8")
    meta = {
        "top_n": args.top_n,
        "safety_risk_threshold": args.safety_risk_threshold,
        "watch_risk_threshold": args.watch_risk_threshold,
        "safe_expansion_threshold": args.safe_expansion_threshold,
        "low_risk_threshold": args.low_risk_threshold,
        "rows": {name: int(len(table)) for name, table in tables.items()},
    }
    (output_dir / "intervention_candidate_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze best-agent intervention candidates from diagnostic scores.")
    parser.add_argument("--risk-scores", type=Path, required=True)
    parser.add_argument("--safe-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safety-risk-threshold", type=float, default=0.70)
    parser.add_argument("--watch-risk-threshold", type=float, default=0.35)
    parser.add_argument("--safe-expansion-threshold", type=float, default=0.35)
    parser.add_argument("--low-risk-threshold", type=float, default=0.20)
    parser.add_argument("--low-worker-citytile-ratio", type=float, default=0.80)
    parser.add_argument("--expansion-min-turn", type=int, default=80)
    parser.add_argument("--expansion-max-turn", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    data = load_scores(args.risk_scores, args.safe_scores)
    cand = data[data["is_candidate"]].copy()
    tables = {
        "safety_gate_candidates": build_safety_candidates(cand, args),
        "expansion_suggestion_candidates": build_expansion_candidates(cand, args),
        "risk_watch_candidates": build_watchlist(cand, args),
    }
    write_outputs(args.output_dir, tables, args)
    print(json.dumps({name: int(len(table)) for name, table in tables.items()}, indent=2))
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
