#!/usr/bin/env python3
"""Dry-run a validated action-gate policy against intervention candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ACTION_COUNT_COLUMNS = {
    "bw": "bw_low_fuel_lt5_actions",
    "bcity": "bcity_adjacent_low_fuel_lt5_actions",
}


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def load_policy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def event_reason(row: pd.Series, rule: dict, count: int) -> str:
    return (
        f"rule={rule['name']}; mode={rule['mode']}; action={rule['action']}; "
        f"count={count}; risk={row['p_loss_10']:.3f}; "
        f"future_loss_10={row['future_team_loss_10']:.0f}; "
        f"min_fuel={row['min_city_fuel_turns']:.2f}; p25_fuel={row['p25_city_fuel_turns']:.2f}"
    )


def apply_policy(candidates: pd.DataFrame, policy: dict) -> pd.DataFrame:
    events = []
    safety = candidates[candidates["candidate_type"].eq("safety_gate")].copy()
    for rule in policy.get("rules", []):
        action = rule.get("action")
        count_col = ACTION_COUNT_COLUMNS.get(action)
        if count_col is None:
            continue
        subset = safety.copy()
        subset = subset[subset["p_loss_10"].ge(float(rule.get("risk_threshold", 1.0)))]
        max_turn = int(rule.get("max_turn", 0) or 0)
        if max_turn > 0:
            subset = subset[subset["turn"].le(max_turn)]
        subset = subset[subset[count_col] > 0]
        for _, row in subset.iterrows():
            count = int(row[count_col])
            event = row.to_dict()
            event.update(
                {
                    "gate_rule": rule["name"],
                    "gate_mode": rule.get("mode", "dry_run"),
                    "gate_action": action,
                    "would_intervene": True,
                    "would_block_action_count": count,
                    "gate_reason": event_reason(row, rule, count),
                }
            )
            events.append(event)
    if not events:
        return pd.DataFrame()
    out = pd.DataFrame(events)
    keep_cols = [
        "gate_rule",
        "gate_mode",
        "gate_action",
        "would_intervene",
        "would_block_action_count",
        "gate_reason",
        "file",
        "opponent",
        "seed",
        "eval_side",
        "team",
        "turn",
        "turn_bucket",
        "p_loss_10",
        "future_team_loss_10",
        "city_tiles",
        "workers",
        "worker_citytile_ratio",
        "min_city_fuel_turns",
        "p25_city_fuel_turns",
        "bcity_actions",
        "bw_actions",
        "bw_low_fuel_lt5_actions",
        "bcity_adjacent_low_fuel_lt5_actions",
    ]
    return out[[column for column in keep_cols if column in out.columns]].sort_values(
        ["gate_rule", "p_loss_10", "future_team_loss_10"], ascending=[True, False, False]
    )


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "gate_rule",
                "gate_mode",
                "event_rows",
                "would_block_actions",
                "loss_rate_10",
                "big_loss_rate_10",
                "mean_future_loss_10",
                "mean_risk",
            ]
        )
    return (
        events.groupby(["gate_rule", "gate_mode"], dropna=False)
        .agg(
            event_rows=("future_team_loss_10", "size"),
            would_block_actions=("would_block_action_count", "sum"),
            loss_rate_10=("future_team_loss_10", lambda s: (s > 0).mean()),
            big_loss_rate_10=("future_team_loss_10", lambda s: (s >= 5).mean()),
            mean_future_loss_10=("future_team_loss_10", "mean"),
            mean_risk=("p_loss_10", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run a gate policy against candidate rows.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)
    candidates = numeric(
        candidates,
        [
            "turn",
            "p_loss_10",
            "future_team_loss_10",
            "city_tiles",
            "workers",
            "worker_citytile_ratio",
            "min_city_fuel_turns",
            "p25_city_fuel_turns",
            "bcity_actions",
            "bw_actions",
            "bw_low_fuel_lt5_actions",
            "bcity_adjacent_low_fuel_lt5_actions",
        ],
    )
    policy = load_policy(args.policy)
    events = apply_policy(candidates, policy)
    summary = summarize(events)
    events.to_csv(args.output_dir / "gate_dry_run_events.csv", index=False, encoding="utf-8")
    summary.to_csv(args.output_dir / "gate_dry_run_summary.csv", index=False, encoding="utf-8")
    meta = {
        "policy": str(args.policy),
        "event_rows": int(len(events)),
        "would_block_actions": int(events["would_block_action_count"].sum()) if len(events) else 0,
        "output": str(args.output_dir),
    }
    (args.output_dir / "gate_dry_run_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
