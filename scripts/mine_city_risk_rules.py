#!/usr/bin/env python3
"""Mine simple city-risk rule candidates from extracted feature rows."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np


def as_int(row: dict, key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def as_float(row: dict, key: str) -> float:
    return float(row.get(key, 0) or 0)


def make_atoms() -> list[tuple[str, Callable[[dict], bool]]]:
    return [
        ("map16", lambda r: as_int(r, "map_size") == 16),
        ("turn>=80", lambda r: as_int(r, "turn") >= 80),
        ("turn>=120", lambda r: as_int(r, "turn") >= 120),
        ("turn>=160", lambda r: as_int(r, "turn") >= 160),
        ("turn>=240", lambda r: as_int(r, "turn") >= 240),
        ("pre_night", lambda r: as_int(r, "pre_night") == 1),
        ("night", lambda r: as_int(r, "is_night") == 1),
        ("pre_or_night", lambda r: as_int(r, "pre_night") == 1 or as_int(r, "is_night") == 1),
        ("fuel<3", lambda r: as_float(r, "fuel_turns") < 3),
        ("fuel<5", lambda r: as_float(r, "fuel_turns") < 5),
        ("fuel<7", lambda r: as_float(r, "fuel_turns") < 7),
        ("fuel<10", lambda r: as_float(r, "fuel_turns") < 10),
        ("fuel<12", lambda r: as_float(r, "fuel_turns") < 12),
        ("size>=5", lambda r: as_int(r, "city_size") >= 5),
        ("size>=10", lambda r: as_int(r, "city_size") >= 10),
        ("size>=15", lambda r: as_int(r, "city_size") >= 15),
        ("team_tiles>=20", lambda r: as_int(r, "team_city_tiles") >= 20),
        ("team_tiles>=40", lambda r: as_int(r, "team_city_tiles") >= 40),
        ("team_tiles>=60", lambda r: as_int(r, "team_city_tiles") >= 60),
        ("units>=tiles", lambda r: as_int(r, "team_units") >= as_int(r, "team_city_tiles")),
        ("units>tiles+5", lambda r: as_int(r, "team_units") > as_int(r, "team_city_tiles") + 5),
        ("upkeep_inc", lambda r: as_int(r, "upkeep_increased_next") == 1),
        ("bcity", lambda r: as_int(r, "build_city_actions") > 0),
        ("isolated_bcity", lambda r: as_int(r, "isolated_build_city_actions") > 0),
        ("resource_bcity", lambda r: as_int(r, "resource_near_build_city_actions") > 0),
        ("adjacent_bcity", lambda r: as_int(r, "adjacent_build_city_actions") > 0),
    ]


def summarize(rows: list[dict], selected: list[int], label_key: str, base_rate: float) -> dict:
    if not selected:
        return {}
    losses = [as_float(rows[i], label_key) for i in selected]
    positives = [loss > 0 for loss in losses]
    big = [loss >= 5 for loss in losses]
    precision = sum(positives) / len(selected)
    big_precision = sum(big) / len(selected)
    return {
        "n": len(selected),
        "loss_rate": precision,
        "big_loss_rate": big_precision,
        "mean_loss": sum(losses) / len(losses),
        "lift": precision / base_rate if base_rate else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine simple city risk rule candidates.")
    parser.add_argument("--input", type=Path, default=Path("outputs/risk_feature_logs/city_risk_features_outputs_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/risk_feature_logs/mined_city_risk_rules.csv"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-terms", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=1000)
    args = parser.parse_args()

    label_key = f"future_team_loss_{args.horizon}"
    with args.input.open(encoding="utf-8", newline="") as in_file:
        rows = list(csv.DictReader(in_file))
    if args.map_size:
        rows = [row for row in rows if as_int(row, "map_size") == args.map_size]

    base_rate = sum(as_float(row, label_key) > 0 for row in rows) / len(rows)
    labels = np.array([as_float(row, label_key) for row in rows], dtype=np.float32)
    positives = labels > 0
    big_losses = labels >= 5

    atoms = make_atoms()
    atom_masks = []
    for name, predicate in atoms:
        atom_masks.append((name, np.array([predicate(row) for row in rows], dtype=bool)))

    results = []
    for term_count in range(1, args.max_terms + 1):
        for combo in combinations(atom_masks, term_count):
            names = [item[0] for item in combo]
            # Skip redundant fuel threshold combinations.
            if sum(name.startswith("fuel<") for name in names) > 1:
                continue
            if sum(name.startswith("turn>=") for name in names) > 1:
                continue
            if "pre_or_night" in names and ("pre_night" in names or "night" in names):
                continue
            selected_mask = combo[0][1].copy()
            for _, mask in combo[1:]:
                selected_mask &= mask
            n = int(selected_mask.sum())
            if n < args.min_rows:
                continue
            selected_labels = labels[selected_mask]
            loss_rate = float(positives[selected_mask].mean())
            big_loss_rate = float(big_losses[selected_mask].mean())
            mean_loss = float(selected_labels.mean())
            item = {
                "n": n,
                "loss_rate": loss_rate,
                "big_loss_rate": big_loss_rate,
                "mean_loss": mean_loss,
                "lift": loss_rate / base_rate if base_rate else 0.0,
            }
            item["rule"] = " & ".join(names)
            results.append(item)

    results.sort(key=lambda item: (item["big_loss_rate"], item["loss_rate"], item["n"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=["rule", "n", "loss_rate", "big_loss_rate", "mean_loss", "lift"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow({
                "rule": item["rule"],
                "n": item["n"],
                "loss_rate": f"{item['loss_rate']:.5f}",
                "big_loss_rate": f"{item['big_loss_rate']:.5f}",
                "mean_loss": f"{item['mean_loss']:.5f}",
                "lift": f"{item['lift']:.3f}",
            })
    print(f"rows: {len(rows)}")
    print(f"base loss rate: {base_rate:.5f}")
    print(f"rules: {len(results)}")
    print(f"output: {args.output}")
    print("top rules:")
    for item in results[:20]:
        print(
            f"{item['rule']}: n={item['n']} loss={item['loss_rate']:.3f} "
            f"big={item['big_loss_rate']:.3f} mean={item['mean_loss']:.3f} lift={item['lift']:.2f}"
        )


if __name__ == "__main__":
    main()
