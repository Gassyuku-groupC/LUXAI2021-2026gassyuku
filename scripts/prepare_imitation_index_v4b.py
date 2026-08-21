#!/usr/bin/env python3
"""Prepare a conservative 16x16 safety-correction BC index."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and reweight the HQ imitation index for BC v4b.")
    parser.add_argument("--input", type=Path, default=Path("dataset/processed/imitation_index_hq.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/imitation_index_v4b_16safe.csv"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument(
        "--teacher-player",
        type=int,
        choices=[-1, 0, 1],
        default=-1,
        help="Filter to one teacher player; -1 keeps both players.",
    )
    parser.add_argument("--max-teacher-night-loss", type=int, default=10)
    parser.add_argument("--night-scale", type=float, default=1.25)
    parser.add_argument("--early-scale", type=float, default=1.0)
    parser.add_argument("--mid-scale", type=float, default=1.0)
    parser.add_argument("--late-scale", type=float, default=1.15)
    parser.add_argument("--player1-scale", type=float, default=1.10)
    parser.add_argument("--max-weight", type=float, default=3.5)
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if int(row["width"]) != args.map_size or int(row["height"]) != args.map_size:
                continue
            if args.teacher_player >= 0 and int(row["teacher_player"]) != args.teacher_player:
                continue
            if int(row["max_teacher_night_loss"]) > args.max_teacher_night_loss:
                continue

            weight = float(row["weight"])
            reasons = [row.get("weight_reason", "")]
            state_step = int(row["state_step"])
            if int(row["is_night"]):
                weight *= args.night_scale
                reasons.append(f"v4b_nightx{args.night_scale:g}")
            if state_step < 120 and args.early_scale != 1.0:
                weight *= args.early_scale
                reasons.append(f"v4b_earlyx{args.early_scale:g}")
            if 120 <= state_step < 240 and args.mid_scale != 1.0:
                weight *= args.mid_scale
                reasons.append(f"v4b_midx{args.mid_scale:g}")
            if state_step >= 240:
                weight *= args.late_scale
                reasons.append(f"v4b_latex{args.late_scale:g}")
            if int(row["teacher_player"]) == 1:
                weight *= args.player1_scale
                reasons.append(f"v4b_p1x{args.player1_scale:g}")

            row["weight"] = f"{min(weight, args.max_weight):.4f}"
            row["weight_reason"] = "+".join(part for part in reasons if part)
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_weight = sum(float(row["weight"]) for row in rows)
    night_rows = sum(int(row["is_night"]) for row in rows)
    player1_rows = sum(int(row["teacher_player"]) == 1 for row in rows)
    print(f"rows: {len(rows)}")
    print(f"night rows: {night_rows}")
    print(f"player1 rows: {player1_rows}")
    print(f"mean weight: {total_weight / len(rows):.3f}" if rows else "mean weight: n/a")
    print(f"index: {args.output}")


if __name__ == "__main__":
    main()
