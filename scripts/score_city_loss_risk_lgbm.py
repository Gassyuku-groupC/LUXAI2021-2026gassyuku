#!/usr/bin/env python3
"""Apply a LightGBM city-collapse risk scorer to strategy features."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

from train_city_loss_risk_scorer import as_float


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


def write_report(path: Path, frame: pd.DataFrame, risk_threshold: float) -> None:
    rows = []
    for (team, bucket), group in frame.groupby(["team", "turn_bucket"], sort=True):
        rows.append({
            "side": f"p{int(team)}",
            "turn_bucket": bucket,
            "n": len(group),
            "mean_risk": group["p_loss_10"].mean(),
            "alert_rate": (group["p_loss_10"] >= risk_threshold).mean(),
            "actual_loss_rate": (group["future_team_loss_10"] > 0).mean() if "future_team_loss_10" in group else 0.0,
            "big_loss_rate": (group["future_team_loss_10"] >= 5).mean() if "future_team_loss_10" in group else 0.0,
            "mean_city_tiles": group["city_tiles"].mean(),
            "mean_min_fuel": group["min_city_fuel_turns"].mean(),
            "mean_p25_fuel": group["p25_city_fuel_turns"].mean(),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score strategy features with a LightGBM city-loss risk model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/diagnostic_layer/risk_scorer_lgbm_v1_top12_16/risk_scorer_lgbm.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    args = parser.parse_args()

    checkpoint = joblib.load(args.model)
    model = checkpoint["model"]
    features = checkpoint["features"]
    frame = pd.read_csv(args.input)
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = 0.0
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    probs = model.predict_proba(frame[features])[:, 1]
    frame["p_loss_10"] = probs
    frame["risk_alert"] = (frame["p_loss_10"] >= args.risk_threshold).astype(int)
    frame["turn_bucket"] = frame["turn"].fillna(0).astype(int).map(turn_bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    write_report(args.report, frame, args.risk_threshold)
    meta = {
        "model": str(args.model),
        "input": str(args.input),
        "output": str(args.output),
        "report": str(args.report),
        "rows": int(len(frame)),
        "risk_threshold": args.risk_threshold,
    }
    args.report.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"rows: {len(frame)}")
    print(f"scored: {args.output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
