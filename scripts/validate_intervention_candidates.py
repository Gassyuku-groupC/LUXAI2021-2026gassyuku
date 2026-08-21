#!/usr/bin/env python3
"""Validate diagnostic intervention candidates and emit a conservative gate spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PATTERNS = [
    {
        "name": "bw_low_fuel_lt5_high_risk",
        "action": "bw",
        "gate_candidate": True,
        "description": "Block build-worker only when the target city fuel buffer is below 5 turns and risk is high.",
        "mask": lambda df: (df["bw_low_fuel_lt5_actions"] > 0),
    },
    {
        "name": "bcity_adjacent_low_fuel_lt5_high_risk",
        "action": "bcity",
        "gate_candidate": True,
        "description": "Block build-city only when it connects to a city with below 5 fuel turns and risk is high.",
        "mask": lambda df: (df["bcity_adjacent_low_fuel_lt5_actions"] > 0),
    },
    {
        "name": "any_bw_high_risk",
        "action": "bw",
        "gate_candidate": False,
        "description": "Watch all build-worker actions at high predicted city-loss risk.",
        "mask": lambda df: (df["bw_actions"] > 0),
    },
    {
        "name": "any_bcity_high_risk",
        "action": "bcity",
        "gate_candidate": False,
        "description": "Watch all build-city actions at high predicted city-loss risk.",
        "mask": lambda df: (df["bcity_actions"] > 0),
    },
]


def to_number(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def load_candidates(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    numeric = [
        "priority_score",
        "turn",
        "eval_side",
        "team",
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
    return to_number(data, numeric)


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(group_cols, dropna=False)
        .agg(
            rows=("future_team_loss_10", "size"),
            loss_rate_10=("future_team_loss_10", lambda s: (s > 0).mean()),
            big_loss_rate_10=("future_team_loss_10", lambda s: (s >= 5).mean()),
            mean_future_loss_10=("future_team_loss_10", "mean"),
            median_risk=("p_loss_10", "median"),
            mean_risk=("p_loss_10", "mean"),
            mean_min_fuel=("min_city_fuel_turns", "mean"),
            mean_p25_fuel=("p25_city_fuel_turns", "mean"),
            mean_city_tiles=("city_tiles", "mean"),
            mean_workers=("workers", "mean"),
        )
        .reset_index()
        .sort_values(["loss_rate_10", "big_loss_rate_10", "rows"], ascending=[False, False, False])
    )


def validate_safety(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    safety = data[data["candidate_type"].eq("safety_gate")].copy()
    safety = safety[safety["p_loss_10"].ge(args.safety_risk_threshold)]
    pattern_rows = []
    gate_rules = []
    for pattern in PATTERNS:
        subset = safety[pattern["mask"](safety)].copy()
        if args.max_gate_turn > 0:
            subset = subset[subset["turn"].le(args.max_gate_turn)]
        rows = int(len(subset))
        if rows == 0:
            metrics = {
                "pattern": pattern["name"],
                "action": pattern["action"],
                "rows": 0,
                "loss_rate_10": 0.0,
                "big_loss_rate_10": 0.0,
                "mean_future_loss_10": 0.0,
                "median_risk": 0.0,
                "approved_for_gate": False,
                "dry_run_ready": False,
                "gate_mode": "log_only",
            }
        else:
            loss_rate = float((subset["future_team_loss_10"] > 0).mean())
            big_loss_rate = float((subset["future_team_loss_10"] >= 5).mean())
            approved = (
                pattern["gate_candidate"]
                and
                rows >= args.min_gate_rows
                and loss_rate >= args.min_loss_rate
                and big_loss_rate >= args.min_big_loss_rate
            )
            dry_run_ready = (
                pattern["gate_candidate"]
                and rows >= args.min_gate_rows
                and loss_rate >= args.min_dry_run_loss_rate
            )
            metrics = {
                "pattern": pattern["name"],
                "action": pattern["action"],
                "gate_candidate": pattern["gate_candidate"],
                "rows": rows,
                "loss_rate_10": loss_rate,
                "big_loss_rate_10": big_loss_rate,
                "mean_future_loss_10": float(subset["future_team_loss_10"].mean()),
                "median_risk": float(subset["p_loss_10"].median()),
                "approved_for_gate": approved,
                "dry_run_ready": dry_run_ready,
                "gate_mode": "block" if approved else ("dry_run" if dry_run_ready else "log_only"),
            }
            if approved or dry_run_ready:
                gate_rules.append(
                    {
                        "name": pattern["name"],
                        "mode": "block" if approved else "dry_run",
                        "action": pattern["action"],
                        "risk_threshold": args.safety_risk_threshold,
                        "max_turn": args.max_gate_turn,
                        "replacement_preference": ["research_or_idle", "keep_original_if_no_safe_replacement"],
                        "description": pattern["description"],
                        "validation": {
                            "rows": rows,
                            "loss_rate_10": loss_rate,
                            "big_loss_rate_10": big_loss_rate,
                            "mean_future_loss_10": float(subset["future_team_loss_10"].mean()),
                        },
                    }
                )
        pattern_rows.append(metrics)
    by_bucket = summarize(safety, ["turn_bucket", "eval_side", "opponent"])
    return pd.DataFrame(pattern_rows), by_bucket, gate_rules


def validate_expansion(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    expansion = data[data["candidate_type"].eq("expansion_suggestion")].copy()
    if args.max_expansion_turn > 0:
        expansion = expansion[expansion["turn"].le(args.max_expansion_turn)]
    expansion["low_capacity_ratio"] = expansion["worker_citytile_ratio"] < args.low_worker_citytile_ratio
    expansion["healthy_fuel_buffer"] = expansion["p25_city_fuel_turns"] >= args.min_expansion_p25_fuel
    expansion["validated_missed_scale"] = expansion["low_capacity_ratio"] & expansion["healthy_fuel_buffer"]
    overall = pd.DataFrame(
        [
            {
                "rows": int(len(expansion)),
                "validated_missed_scale_rate": float(expansion["validated_missed_scale"].mean()) if len(expansion) else 0.0,
                "low_capacity_ratio_rate": float(expansion["low_capacity_ratio"].mean()) if len(expansion) else 0.0,
                "healthy_fuel_buffer_rate": float(expansion["healthy_fuel_buffer"].mean()) if len(expansion) else 0.0,
                "mean_safe_expansion": float(expansion["p_safe_expansion"].mean()) if len(expansion) else 0.0,
                "mean_risk": float(expansion["p_loss_10"].mean()) if len(expansion) else 0.0,
                "mean_worker_citytile_ratio": float(expansion["worker_citytile_ratio"].mean()) if len(expansion) else 0.0,
                "mean_p25_fuel": float(expansion["p25_city_fuel_turns"].mean()) if len(expansion) else 0.0,
            }
        ]
    )
    by_bucket = (
        expansion.groupby(["turn_bucket", "eval_side", "opponent"], dropna=False)
        .agg(
            rows=("p_safe_expansion", "size"),
            validated_missed_scale_rate=("validated_missed_scale", "mean"),
            mean_safe_expansion=("p_safe_expansion", "mean"),
            mean_risk=("p_loss_10", "mean"),
            mean_worker_citytile_ratio=("worker_citytile_ratio", "mean"),
            mean_p25_fuel=("p25_city_fuel_turns", "mean"),
            mean_city_tiles=("city_tiles", "mean"),
            mean_workers=("workers", "mean"),
        )
        .reset_index()
        .sort_values(["validated_missed_scale_rate", "rows"], ascending=[False, False])
        if len(expansion)
        else pd.DataFrame()
    )
    return overall, by_bucket


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    limited = frame.head(max_rows).copy()
    for column in limited.columns:
        if pd.api.types.is_float_dtype(limited[column]):
            limited[column] = limited[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in limited.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in limited.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in limited.columns) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    safety_patterns: pd.DataFrame,
    safety_by_bucket: pd.DataFrame,
    expansion_overall: pd.DataFrame,
    expansion_by_bucket: pd.DataFrame,
    gate_rules: list[dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Intervention Validation Report",
        "",
        "## Thresholds",
        "",
        f"- safety_risk_threshold: `{args.safety_risk_threshold}`",
        f"- min_gate_rows: `{args.min_gate_rows}`",
        f"- min_loss_rate: `{args.min_loss_rate}`",
        f"- min_big_loss_rate: `{args.min_big_loss_rate}`",
        f"- max_gate_turn: `{args.max_gate_turn}`",
        "",
        "## Safety Gate Pattern Validation",
        "",
        markdown_table(safety_patterns),
        "",
        "## Safety Hotspots",
        "",
        markdown_table(safety_by_bucket),
        "",
        "## Expansion Validation",
        "",
        markdown_table(expansion_overall),
        "",
        "## Expansion Hotspots",
        "",
        markdown_table(expansion_by_bucket),
        "",
        "## Gate Decision",
        "",
    ]
    if gate_rules:
        lines.append("The following rules passed validation and are eligible for dry-run safety-gate testing:")
        lines.append("")
        for rule in gate_rules:
            validation = rule["validation"]
            lines.append(
                f"- `{rule['name']}`: rows={validation['rows']}, "
                f"loss_rate_10={validation['loss_rate_10']:.3f}, "
                f"big_loss_rate_10={validation['big_loss_rate_10']:.3f}"
            )
    else:
        lines.append("No rule passed the conservative gate thresholds. Keep all rules log-only.")
    (output_dir / "intervention_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate intervention candidates and emit gate policy spec.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safety-risk-threshold", type=float, default=0.70)
    parser.add_argument("--min-gate-rows", type=int, default=8)
    parser.add_argument("--min-dry-run-loss-rate", type=float, default=0.60)
    parser.add_argument("--min-loss-rate", type=float, default=0.75)
    parser.add_argument("--min-big-loss-rate", type=float, default=0.35)
    parser.add_argument("--max-gate-turn", type=int, default=160)
    parser.add_argument("--max-expansion-turn", type=int, default=320)
    parser.add_argument("--low-worker-citytile-ratio", type=float, default=0.80)
    parser.add_argument("--min-expansion-p25-fuel", type=float, default=10.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_candidates(args.candidates)
    safety_patterns, safety_by_bucket, gate_rules = validate_safety(data, args)
    expansion_overall, expansion_by_bucket = validate_expansion(data, args)

    safety_patterns.to_csv(args.output_dir / "safety_pattern_validation.csv", index=False, encoding="utf-8")
    safety_by_bucket.to_csv(args.output_dir / "safety_hotspots_by_bucket.csv", index=False, encoding="utf-8")
    expansion_overall.to_csv(args.output_dir / "expansion_validation_overall.csv", index=False, encoding="utf-8")
    expansion_by_bucket.to_csv(args.output_dir / "expansion_hotspots_by_bucket.csv", index=False, encoding="utf-8")

    gate_spec = {
        "enabled": False,
        "default_mode": "dry_run",
        "source_candidates": str(args.candidates),
        "rules": gate_rules,
        "notes": [
            "Keep enabled=false until the same rules validate across more random seeds.",
            "Prefer blocking build-worker before blocking build-city; bcity can be a lifeboat.",
            "Replacement should be non-invasive: city research/idle for bw, keep original if no safe fallback is known.",
        ],
    }
    (args.output_dir / "gate_policy_spec.json").write_text(json.dumps(gate_spec, indent=2), encoding="utf-8")
    write_report(output_dir=args.output_dir, safety_patterns=safety_patterns, safety_by_bucket=safety_by_bucket,
                 expansion_overall=expansion_overall, expansion_by_bucket=expansion_by_bucket,
                 gate_rules=gate_rules, args=args)
    print(json.dumps({
        "candidate_rows": int(len(data)),
        "dry_run_rules": sum(1 for rule in gate_rules if rule["mode"] == "dry_run"),
        "block_rules": sum(1 for rule in gate_rules if rule["mode"] == "block"),
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
