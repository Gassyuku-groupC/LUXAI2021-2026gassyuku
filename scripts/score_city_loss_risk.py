#!/usr/bin/env python3
"""Apply a trained city-collapse risk scorer to strategy features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from train_city_loss_risk_scorer import RiskMLP, as_float  # type: ignore


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


def side_from_row(row: dict) -> str:
    return f"p{int(as_float(row, 'team'))}"


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    model = RiskMLP(len(checkpoint["features"]), int(checkpoint["hidden"]), float(checkpoint["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def feature_tensor(row: dict, features: list[str], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    values = torch.tensor([as_float(row, feature) for feature in features], dtype=torch.float32)
    return (values - mean) / std


def write_report(path: Path, rows: list[dict], risk_threshold: float) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(side_from_row(row), turn_bucket(int(as_float(row, "turn"))))].append(row)

    lines = [
        "side,turn_bucket,n,mean_risk,alert_rate,actual_loss_rate,big_loss_rate,mean_city_tiles,mean_min_fuel,mean_p25_fuel"
    ]
    for key, group in sorted(groups.items()):
        n = len(group)
        mean_risk = sum(float(row["p_loss_10"]) for row in group) / n
        alert_rate = sum(1 for row in group if float(row["p_loss_10"]) >= risk_threshold) / n
        actual_loss_rate = sum(1 for row in group if as_float(row, "future_team_loss_10") > 0) / n
        big_loss_rate = sum(1 for row in group if as_float(row, "future_team_loss_10") >= 5) / n
        mean_city = sum(as_float(row, "city_tiles") for row in group) / n
        mean_min = sum(as_float(row, "min_city_fuel_turns") for row in group) / n
        mean_p25 = sum(as_float(row, "p25_city_fuel_turns") for row in group) / n
        lines.append(
            f"{key[0]},{key[1]},{n},{mean_risk:.4f},{alert_rate:.4f},"
            f"{actual_loss_rate:.4f},{big_loss_rate:.4f},{mean_city:.4f},{mean_min:.4f},{mean_p25:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score strategy features with a city-loss risk model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/diagnostic_layer/risk_scorer_v1/risk_scorer.pt"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    model, checkpoint = load_model(args.model)
    features = list(checkpoint["features"])
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    scored_rows = []
    with args.input.open(encoding="utf-8", newline="") as in_file, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as out_file:
        reader = csv.DictReader(in_file)
        fieldnames = list(reader.fieldnames or []) + ["p_loss_10", "risk_alert"]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        batch_rows = []
        batch_x = []
        total = 0
        for row in reader:
            batch_rows.append(row)
            batch_x.append(feature_tensor(row, features, mean, std))
            if len(batch_x) >= 4096:
                x = torch.stack(batch_x)
                with torch.no_grad():
                    probs = torch.sigmoid(model(x)).tolist()
                for item, prob in zip(batch_rows, probs):
                    item = dict(item)
                    item["p_loss_10"] = f"{prob:.6f}"
                    item["risk_alert"] = "1" if prob >= args.risk_threshold else "0"
                    writer.writerow(item)
                    scored_rows.append(item)
                total += len(batch_x)
                batch_rows, batch_x = [], []
                if args.max_rows and total >= args.max_rows:
                    break
        if batch_x and not (args.max_rows and total >= args.max_rows):
            x = torch.stack(batch_x)
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).tolist()
            for item, prob in zip(batch_rows, probs):
                item = dict(item)
                item["p_loss_10"] = f"{prob:.6f}"
                item["risk_alert"] = "1" if prob >= args.risk_threshold else "0"
                writer.writerow(item)
                scored_rows.append(item)

    write_report(args.report, scored_rows, args.risk_threshold)
    meta = {
        "model": str(args.model),
        "input": str(args.input),
        "output": str(args.output),
        "report": str(args.report),
        "rows": len(scored_rows),
        "risk_threshold": args.risk_threshold,
    }
    (args.report.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"rows: {len(scored_rows)}")
    print(f"scored: {args.output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
