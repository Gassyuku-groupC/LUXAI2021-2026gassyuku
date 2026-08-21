#!/usr/bin/env python3
"""Combine risk and safe-expansion scores into one evaluation diagnostic report."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


def turn_bucket(turn: int) -> str:
    if turn < 40:
        return "000-039"
    if turn < 80:
        return "040-079"
    if turn < 120:
        return "080-119"
    if turn < 160:
        return "120-159"
    if turn < 240:
        return "160-239"
    if turn < 320:
        return "240-319"
    return "320-360"


def parse_replay_name(path: str) -> tuple[str, str, int | None]:
    name = os.path.basename(str(path))
    match = re.search(r"map_\d+x\d+_vs_(.+)_(\d+)_p([01])\.json$", name)
    if not match:
        return "unknown", "unknown", None
    return match.group(1), match.group(2), int(match.group(3))


def load_scores(risk_path: Path, safe_path: Path) -> pd.DataFrame:
    risk_cols = ["file", "team", "turn", "p_loss_10", "risk_alert"]
    risk = pd.read_csv(risk_path, usecols=lambda col: col in risk_cols)
    safe = pd.read_csv(safe_path)
    data = safe.merge(risk, on=["file", "team", "turn"], how="left")
    parsed = data["file"].map(parse_replay_name)
    data["opponent"] = [item[0] for item in parsed]
    data["seed"] = [item[1] for item in parsed]
    data["eval_side"] = [item[2] for item in parsed]
    data["candidate_side"] = data["eval_side"]
    data["is_candidate"] = data["eval_side"].notna() & (
        data["team"].astype(int) == data["eval_side"].astype(int)
    )
    data["turn_bucket"] = data["turn"].fillna(0).astype(int).map(turn_bucket)
    numeric = [
        "p_loss_10",
        "p_safe_expansion",
        "future_team_loss_10",
        "bcity_actions",
        "bw_actions",
        "bw_low_fuel_lt5_actions",
        "bcity_adjacent_low_fuel_lt5_actions",
        "city_tiles",
        "workers",
        "worker_citytile_ratio",
        "min_city_fuel_turns",
        "p25_city_fuel_turns",
        "fuel_turns_total",
        "final_city_tiles",
    ]
    for column in numeric:
        if column not in data.columns:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    return data


def group_summary(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cand = data[data["is_candidate"]].copy()
    cand["risk_alert_bool"] = cand["p_loss_10"] >= args.risk_threshold
    cand["safe_opp_bool"] = cand["p_safe_expansion"] >= args.safe_threshold
    cand["missed_safe_opp_bool"] = cand["safe_opp_bool"] & (cand["bcity_actions"] <= 0) & (
        cand["p_loss_10"] < args.low_risk_threshold
    )
    cand["dangerous_expansion_bool"] = (cand["p_loss_10"] >= args.risk_threshold) & (
        (cand["bcity_actions"] > 0) | (cand["bw_actions"] > 0)
    )
    grouped = cand.groupby(["opponent", "eval_side", "turn_bucket"], sort=True)
    return grouped.agg(
        n=("p_loss_10", "size"),
        mean_risk=("p_loss_10", "mean"),
        risk_alert_rate=("risk_alert_bool", "mean"),
        actual_loss_rate=("future_team_loss_10", lambda s: (s > 0).mean()),
        big_loss_rate=("future_team_loss_10", lambda s: (s >= 5).mean()),
        mean_safe_expansion=("p_safe_expansion", "mean"),
        safe_opportunity_rate=("safe_opp_bool", "mean"),
        missed_safe_opportunity_rate=("missed_safe_opp_bool", "mean"),
        dangerous_action_rate=("dangerous_expansion_bool", "mean"),
        actual_bcity_rate=("bcity_actions", lambda s: (s > 0).mean()),
        mean_city_tiles=("city_tiles", "mean"),
        mean_workers=("workers", "mean"),
        mean_worker_citytile_ratio=("worker_citytile_ratio", "mean"),
        mean_min_fuel=("min_city_fuel_turns", "mean"),
        mean_p25_fuel=("p25_city_fuel_turns", "mean"),
        mean_final_city_tiles=("final_city_tiles", "mean"),
    ).reset_index()


def top_rows(data: pd.DataFrame, mask: pd.Series, sort_cols: list[str], ascending: list[bool], n: int) -> pd.DataFrame:
    cols = [
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
        "bcity_actions",
        "bw_actions",
        "bw_low_fuel_lt5_actions",
        "bcity_adjacent_low_fuel_lt5_actions",
    ]
    out = data[mask].copy()
    if out.empty:
        return out[cols]
    out["file"] = out["file"].map(lambda value: os.path.basename(str(value)))
    return out.sort_values(sort_cols, ascending=ascending).head(n)[cols]


def write_markdown(path: Path, summary: pd.DataFrame, tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        limited = frame.copy()
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

    lines = [
        "# Evaluation Diagnostic Report",
        "",
        f"- risk_threshold: `{args.risk_threshold}`",
        f"- low_risk_threshold: `{args.low_risk_threshold}`",
        f"- safe_threshold: `{args.safe_threshold}`",
        "",
        "## Phase Summary",
        "",
        markdown_table(summary),
    ]
    for title, table in tables.items():
        lines.extend(["", f"## {title}", ""])
        lines.append(markdown_table(table))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a combined risk/safe-expansion diagnostic report.")
    parser.add_argument("--risk-scores", type=Path, required=True)
    parser.add_argument("--safe-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument("--low-risk-threshold", type=float, default=0.20)
    parser.add_argument("--safe-threshold", type=float, default=0.35)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    data = load_scores(args.risk_scores, args.safe_scores)
    cand = data[data["is_candidate"]].copy()
    summary = group_summary(data, args)
    missed_mask = (
        cand["p_safe_expansion"].ge(args.safe_threshold)
        & cand["p_loss_10"].lt(args.low_risk_threshold)
        & cand["bcity_actions"].le(0)
    )
    dangerous_mask = (
        cand["p_loss_10"].ge(args.risk_threshold)
        & ((cand["bcity_actions"] > 0) | (cand["bw_actions"] > 0))
    )
    high_risk_mask = cand["p_loss_10"].ge(args.risk_threshold)

    tables = {
        "Top Missed Safe Expansion": top_rows(
            cand,
            missed_mask,
            ["p_safe_expansion", "p_loss_10"],
            [False, True],
            args.top_n,
        ),
        "Top High Risk Candidate Turns": top_rows(
            cand,
            high_risk_mask,
            ["p_loss_10", "future_team_loss_10"],
            [False, False],
            args.top_n,
        ),
        "Dangerous Build/Worker Actions": top_rows(
            cand,
            dangerous_mask,
            ["p_loss_10", "future_team_loss_10"],
            [False, False],
            args.top_n,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "diagnostic_phase_summary.csv", index=False, encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(args.output_dir / (name.lower().replace(" ", "_").replace("/", "_") + ".csv"), index=False, encoding="utf-8")
    write_markdown(args.output_dir / "diagnostic_report.md", summary, tables, args)
    meta = {
        "risk_scores": str(args.risk_scores),
        "safe_scores": str(args.safe_scores),
        "rows": int(len(data)),
        "candidate_rows": int(len(cand)),
        "missed_safe_expansion_rows": int(missed_mask.sum()),
        "high_risk_rows": int(high_risk_mask.sum()),
        "dangerous_action_rows": int(dangerous_mask.sum()),
    }
    (args.output_dir / "diagnostic_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"report: {args.output_dir / 'diagnostic_report.md'}")


if __name__ == "__main__":
    main()
