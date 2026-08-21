#!/usr/bin/env python3
"""Summarize one Lux agent experiment against a baseline report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_groups(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = metrics.get("groups", {})
    return groups if isinstance(groups, dict) else {}


def replay_details(metrics: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = metrics.get("details", [])
    out = {}
    for row in rows if isinstance(rows, list) else []:
        key = replay_key(row)
        if key:
            out[key] = row
    return out


def replay_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    file_name = Path(str(row.get("file") or row.get("replay") or row.get("path") or "")).name
    match = re.search(r"map_\d+x\d+_vs_(.+)_(\d+)_p([01])(?:\.json)?$", file_name)
    if match:
        return match.group(1), match.group(2), f"p{match.group(3)}"
    opponent = row.get("opponent")
    seed = row.get("seed")
    player = row.get("player")
    if opponent is None or seed is None or player is None:
        return None
    return str(opponent), str(seed), f"p{player}"


def num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def group_row(name: str, candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "group": name,
        "candidate_games": int(num(candidate.get("games"))),
        "baseline_games": int(num(baseline.get("games"))),
        "candidate_win_rate": num(candidate.get("win_rate")),
        "baseline_win_rate": num(baseline.get("win_rate")),
        "win_rate_delta": num(candidate.get("win_rate")) - num(baseline.get("win_rate")),
        "candidate_survival": num(candidate.get("effective_survival_rate")),
        "baseline_survival": num(baseline.get("effective_survival_rate")),
        "survival_delta": num(candidate.get("effective_survival_rate")) - num(baseline.get("effective_survival_rate")),
        "candidate_city_tiles": num(candidate.get("mean_city_tiles")),
        "baseline_city_tiles": num(baseline.get("mean_city_tiles")),
        "city_tiles_delta": num(candidate.get("mean_city_tiles")) - num(baseline.get("mean_city_tiles")),
        "candidate_worst_night_loss": num(candidate.get("worst_night_city_loss")),
        "baseline_worst_night_loss": num(baseline.get("worst_night_city_loss")),
        "worst_night_loss_delta": num(candidate.get("worst_night_city_loss")) - num(baseline.get("worst_night_city_loss")),
        "candidate_uranium_rate": num(candidate.get("uranium_rate")),
        "baseline_uranium_rate": num(baseline.get("uranium_rate")),
        "uranium_rate_delta": num(candidate.get("uranium_rate")) - num(baseline.get("uranium_rate")),
        "candidate_side_gap": num((candidate.get("sides") or {}).get("city_gap")),
        "baseline_side_gap": num((baseline.get("sides") or {}).get("city_gap")),
        "side_gap_delta": num((candidate.get("sides") or {}).get("city_gap")) - num((baseline.get("sides") or {}).get("city_gap")),
    }


def risk_summary(report: dict[str, Any]) -> dict[str, Any]:
    actionable = report.get("actionable_thresholded_per_night_top1", {}) or {}
    combined = report.get("combined_per_night_top1", {}) or {}
    return {
        "late_rows": int(num(report.get("late_rows"))),
        "suggestion_rows": int(num(report.get("suggestion_rows"))),
        "combined_hit_rate": num(combined.get("combined_hit_rate")),
        "actionable_hit_rate": num(actionable.get("actionable_combined_hit_rate")),
        "late_actionable_hit_rate": num(actionable.get("late_actionable_hit_rate")),
        "suggestion_actionable_hit_rate": num(actionable.get("suggestion_actionable_hit_rate")),
        "actionable_misses": int(num(actionable.get("actionable_misses"))),
    }


def detail_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "games": 0,
            "mean_city_tiles": 0.0,
            "mean_units": 0.0,
            "mean_fuel": 0.0,
            "mean_upkeep": 0.0,
            "mean_night_loss": 0.0,
            "max_night_loss": 0.0,
            "loss_gt_10": 0,
            "loss_gt_20": 0,
            "loss_gt_40": 0,
            "loss_gt_60": 0,
            "effective_survival_rate": 0.0,
        }
    losses = [num(row.get("max_night_city_loss")) for row in rows]
    return {
        "games": len(rows),
        "mean_city_tiles": sum(num(row.get("city_tiles")) for row in rows) / len(rows),
        "mean_units": sum(num(row.get("units")) for row in rows) / len(rows),
        "mean_fuel": sum(num(row.get("fuel")) for row in rows) / len(rows),
        "mean_upkeep": sum(num(row.get("upkeep")) for row in rows) / len(rows),
        "mean_night_loss": sum(losses) / len(losses),
        "max_night_loss": max(losses),
        "loss_gt_10": sum(loss > 10 for loss in losses),
        "loss_gt_20": sum(loss > 20 for loss in losses),
        "loss_gt_40": sum(loss > 40 for loss in losses),
        "loss_gt_60": sum(loss > 60 for loss in losses),
        "effective_survival_rate": sum(1 for row in rows if row.get("effective_survival")) / len(rows),
    }


def common_replay_comparison(candidate_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    candidate_details = replay_details(candidate_metrics)
    baseline_details = replay_details(baseline_metrics)
    common_keys = sorted(set(candidate_details) & set(baseline_details))
    candidate_rows = [candidate_details[key] for key in common_keys]
    baseline_rows = [baseline_details[key] for key in common_keys]
    candidate = detail_summary(candidate_rows)
    baseline = detail_summary(baseline_rows)
    delta = {
        key: num(candidate.get(key)) - num(baseline.get(key))
        for key in set(candidate) | set(baseline)
    }
    return {
        "common_games": len(common_keys),
        "candidate": candidate,
        "baseline": baseline,
        "delta": delta,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    verdict = summary.get("verdict", {})
    lines = [
        "# Experiment Summary",
        "",
        f"- experiment: `{summary.get('experiment_name', '')}`",
        f"- candidate: `{summary.get('candidate_agent', '')}`",
        f"- baseline: `{summary.get('baseline_agent', '')}`",
        f"- verdict: **{verdict.get('label', 'unknown')}**",
        f"- reason: {verdict.get('reason', '')}",
        "",
        "## Promotion Groups",
        "",
        table(
            summary.get("group_comparison", []),
            [
                "group",
                "candidate_games",
                "candidate_win_rate",
                "win_rate_delta",
                "candidate_city_tiles",
                "city_tiles_delta",
                "candidate_worst_night_loss",
                "worst_night_loss_delta",
                "candidate_side_gap",
                "side_gap_delta",
            ],
        ),
        "",
        "## Risk Summary",
        "",
        table(
            [
                {"name": "candidate", **summary.get("candidate_risk", {})},
                {"name": "baseline", **summary.get("baseline_risk", {})},
                {"name": "delta", **summary.get("risk_delta", {})},
            ],
            [
                "name",
                "late_rows",
                "suggestion_rows",
                "combined_hit_rate",
                "actionable_hit_rate",
                "late_actionable_hit_rate",
                "suggestion_actionable_hit_rate",
                "actionable_misses",
            ],
        ),
    ]
    common = summary.get("common_replay_comparison", {})
    if common:
        lines.extend([
            "",
            "## Common Replay A/B",
            "",
            table(
                [
                    {"name": "candidate", **common.get("candidate", {})},
                    {"name": "baseline", **common.get("baseline", {})},
                    {"name": "delta", **common.get("delta", {})},
                ],
                [
                    "name",
                    "games",
                    "mean_city_tiles",
                    "mean_units",
                    "mean_fuel",
                    "mean_upkeep",
                    "mean_night_loss",
                    "max_night_loss",
                    "loss_gt_20",
                    "loss_gt_40",
                    "loss_gt_60",
                    "effective_survival_rate",
                ],
            ),
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_verdict(rows: list[dict[str, Any]], risk_delta: dict[str, Any]) -> dict[str, str]:
    if not rows:
        return {"label": "needs_eval", "reason": "No comparable promotion groups were found."}
    worse_loss = max(row["worst_night_loss_delta"] for row in rows)
    worse_city = min(row["city_tiles_delta"] for row in rows)
    worse_win = min(row["win_rate_delta"] for row in rows)
    suggestion_delta = num(risk_delta.get("suggestion_rows"))
    if worse_win < -0.01:
        return {"label": "reject", "reason": "Win rate regressed on at least one group."}
    if worse_loss > 20:
        return {"label": "reject", "reason": "Worst night city loss increased by more than 20."}
    if worse_city < -3 and suggestion_delta >= 0:
        return {"label": "reject", "reason": "Scale dropped without reducing suggestion-risk rows."}
    if worse_loss <= 0 and worse_city >= -1:
        return {"label": "candidate", "reason": "Risk did not worsen and city scale was preserved."}
    return {"label": "watch", "reason": "Mixed result; inspect per-opponent and replay diagnostics before promotion."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an automated Lux experiment.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--candidate-agent", type=Path, required=True)
    parser.add_argument("--baseline-agent", type=Path, required=True)
    parser.add_argument("--candidate-promotion", type=Path, required=True)
    parser.add_argument("--baseline-promotion", type=Path, required=True)
    parser.add_argument("--candidate-risk", type=Path, default=Path(""))
    parser.add_argument("--baseline-risk", type=Path, default=Path(""))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_metrics = load_json(args.candidate_promotion)
    baseline_metrics = load_json(args.baseline_promotion)
    candidate_groups = flatten_groups(candidate_metrics)
    baseline_groups = flatten_groups(baseline_metrics)
    common = sorted(set(candidate_groups) & set(baseline_groups))
    rows = [group_row(name, candidate_groups[name], baseline_groups[name]) for name in common]
    common_replays = common_replay_comparison(candidate_metrics, baseline_metrics)

    cand_risk = risk_summary(load_json(args.candidate_risk)) if str(args.candidate_risk) else {}
    base_risk = risk_summary(load_json(args.baseline_risk)) if str(args.baseline_risk) else {}
    risk_delta = {
        key: num(cand_risk.get(key)) - num(base_risk.get(key))
        for key in set(cand_risk) | set(base_risk)
    }

    summary = {
        "experiment_name": args.experiment_name,
        "candidate_agent": str(args.candidate_agent),
        "baseline_agent": str(args.baseline_agent),
        "candidate_promotion": str(args.candidate_promotion),
        "baseline_promotion": str(args.baseline_promotion),
        "candidate_risk_report": str(args.candidate_risk),
        "baseline_risk_report": str(args.baseline_risk),
        "group_comparison": rows,
        "candidate_risk": cand_risk,
        "baseline_risk": base_risk,
        "risk_delta": risk_delta,
        "common_replay_comparison": common_replays,
        "verdict": make_verdict(rows, risk_delta),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "experiment_summary.md", summary)
    print(json.dumps(summary["verdict"], indent=2))
    print(f"summary: {args.output_dir / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
