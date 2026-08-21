#!/usr/bin/env python3
"""Apply a LightGBM safe-expansion scorer to strategy features."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import joblib
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


def eval_side(path: str) -> int | None:
    match = re.search(r"_p([01])\.json$", os.path.basename(str(path)))
    if not match:
        return None
    return int(match.group(1))


def write_report(path: Path, frame: pd.DataFrame, threshold: float) -> None:
    rows = []
    for keys, group in frame.groupby(["opponent_name", "candidate_side", "turn_bucket"], sort=True):
        opponent, side, bucket = keys
        if len(group) == 0:
            continue
        rows.append({
            "opponent": opponent,
            "side": f"p{int(side)}",
            "turn_bucket": bucket,
            "n": len(group),
            "mean_safe_expansion": group["p_safe_expansion"].mean(),
            "opportunity_rate": (group["p_safe_expansion"] >= threshold).mean(),
            "actual_bcity_rate": (group["bcity_actions"] > 0).mean(),
            "missed_opportunity_rate": ((group["p_safe_expansion"] >= threshold) & (group["bcity_actions"] <= 0)).mean(),
            "mean_city_tiles": group["city_tiles"].mean(),
            "mean_min_fuel": group["min_city_fuel_turns"].mean(),
            "mean_p25_fuel": group["p25_city_fuel_turns"].mean(),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score strategy features with a safe-expansion model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/diagnostic_layer/safe_expansion_lgbm_v1_top12_16/safe_expansion_scorer_lgbm.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--candidate-only", action="store_true")
    args = parser.parse_args()

    checkpoint = joblib.load(args.model)
    model = checkpoint["model"]
    features = checkpoint["features"]
    frame = pd.read_csv(args.input)
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = 0.0
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame["candidate_side"] = frame["file"].map(eval_side)
    if args.candidate_only:
        frame = frame[frame["candidate_side"].notna()].copy()
        frame = frame[frame["team"].astype(int) == frame["candidate_side"].astype(int)].copy()
    frame["p_safe_expansion"] = model.predict_proba(frame[features])[:, 1]
    frame["safe_expansion_opportunity"] = (frame["p_safe_expansion"] >= args.threshold).astype(int)
    frame["turn_bucket"] = frame["turn"].fillna(0).astype(int).map(turn_bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    write_report(args.report, frame, args.threshold)
    meta = {
        "model": str(args.model),
        "input": str(args.input),
        "output": str(args.output),
        "report": str(args.report),
        "rows": int(len(frame)),
        "threshold": args.threshold,
        "candidate_only": bool(args.candidate_only),
    }
    args.report.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"rows: {len(frame)}")
    print(f"scored: {args.output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
