#!/usr/bin/env python3
"""Prepare high-confidence fuel-support movement override candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "team",
    "rank",
    "map_size",
    "turn",
    "cycle_turn",
    "turns_to_night",
    "is_night",
    "unit_x",
    "unit_y",
    "unit_cargo_fuel",
    "move_target_x",
    "move_target_y",
    "city_x",
    "city_y",
    "city_fuel_turns",
    "city_fuel",
    "city_upkeep",
    "city_tiles",
    "team_city_tiles",
    "final_city_tiles",
    "future_team_loss_5",
    "future_team_loss_10",
    "future_team_loss_20",
]


OUTPUT_COLUMNS = [
    "priority_score",
    "candidate_rule",
    "event_type",
    "file",
    "team",
    "turn",
    "cycle_turn",
    "turns_to_night",
    "is_night",
    "unit_id",
    "unit_x",
    "unit_y",
    "unit_cargo_fuel",
    "action",
    "city_id",
    "city_x",
    "city_y",
    "city_fuel_turns",
    "city_tiles",
    "team_city_tiles",
    "future_team_loss_5",
    "future_team_loss_10",
    "future_team_loss_20",
]


def load_events(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            data[column] = 0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    if "action" not in data.columns:
        data["action"] = ""
    data["action"] = data["action"].fillna("")
    return data


def select_candidates(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if data.empty:
        return data
    mask = (
        data["event_type"].eq("missed_adjacent_fuel")
        & data["city_fuel_turns"].lt(args.city_fuel_turns_lt)
        & data["unit_cargo_fuel"].ge(args.min_cargo_fuel)
        & data["turn"].le(args.max_turn)
    )
    if args.require_no_action:
        mask &= data["action"].eq("")
    if args.prefer_pre_night:
        mask &= data["turns_to_night"].between(1, args.turns_to_night_lte)
    candidates = data[mask].copy()
    if candidates.empty:
        return candidates
    candidates["candidate_rule"] = "adjacent_low_fuel_cargo_support_v3"
    candidates["priority_score"] = (
        (args.city_fuel_turns_lt - candidates["city_fuel_turns"]).clip(lower=0) * 2.0
        + (candidates["unit_cargo_fuel"].clip(upper=200) / 100.0)
        + candidates["future_team_loss_10"].clip(upper=10) / 2.0
        + candidates["is_night"] * 0.5
    )
    return candidates.sort_values(
        ["priority_score", "future_team_loss_10", "city_fuel_turns"],
        ascending=[False, False, True],
    )


def summarize(candidates: pd.DataFrame) -> dict:
    if candidates.empty:
        return {
            "rows": 0,
            "future_loss_10_rate": 0.0,
            "future_big_loss_10_rate": 0.0,
            "mean_future_loss_10": 0.0,
        }
    loss_10 = candidates["future_team_loss_10"] > 0
    big_loss_10 = candidates["future_team_loss_10"] >= 5
    grouped = (
        candidates.groupby(["candidate_rule", "is_night"], dropna=False)
        .agg(
            rows=("event_type", "size"),
            future_loss_10_rate=("future_team_loss_10", lambda s: (s > 0).mean()),
            future_big_loss_10_rate=("future_team_loss_10", lambda s: (s >= 5).mean()),
            mean_future_loss_10=("future_team_loss_10", "mean"),
            mean_city_fuel_turns=("city_fuel_turns", "mean"),
            mean_unit_cargo_fuel=("unit_cargo_fuel", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(candidates)),
        "future_loss_10_rate": float(loss_10.mean()),
        "future_big_loss_10_rate": float(big_loss_10.mean()),
        "mean_future_loss_10": float(candidates["future_team_loss_10"].mean()),
        "mean_city_fuel_turns": float(candidates["city_fuel_turns"].mean()),
        "mean_unit_cargo_fuel": float(candidates["unit_cargo_fuel"].mean()),
        "by_rule_phase": grouped,
    }


def write_outputs(candidates: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = candidates.copy()
    if not out.empty:
        out = out[[column for column in OUTPUT_COLUMNS if column in out.columns]]
    out.to_csv(output_dir / "fuel_support_v3_candidate_events.csv", index=False, encoding="utf-8")

    summary = summarize(candidates)
    spec = {
        "enabled": False,
        "default_mode": "dry_run",
        "rule": {
            "name": "adjacent_low_fuel_cargo_support_v3",
            "mode": "dry_run",
            "action": "move_worker_to_adjacent_low_fuel_city_tile",
            "city_fuel_turns_lt": args.city_fuel_turns_lt,
            "min_cargo_fuel": args.min_cargo_fuel,
            "max_turn": args.max_turn,
            "turns_to_night_lte": args.turns_to_night_lte,
            "prefer_pre_night": args.prefer_pre_night,
            "require_no_action": args.require_no_action,
            "replacement_policy": "only consider movement override if destination city tile is free and does not conflict with existing collision resolution",
        },
        "validation": summary,
        "notes": [
            "This is a candidate spec only; no runtime movement override is enabled.",
            "The signal is correlational. Validate across more seeds before any block/override mode.",
            "Do not override build-city, build-worker, or transfer actions with this rule.",
        ],
    }
    (output_dir / "movement_override_candidate_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    pd.DataFrame(summary.get("by_rule_phase", [])).to_csv(
        output_dir / "fuel_support_v3_candidate_summary.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "fuel_support_v3_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fuel-support v3 movement override candidates.")
    parser.add_argument("events", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city-fuel-turns-lt", type=float, default=2.5)
    parser.add_argument("--min-cargo-fuel", type=float, default=80.0)
    parser.add_argument("--max-turn", type=int, default=160)
    parser.add_argument("--turns-to-night-lte", type=int, default=3)
    parser.add_argument("--prefer-pre-night", action="store_true")
    parser.add_argument("--require-no-action", action="store_true")
    args = parser.parse_args()

    data = load_events(args.events)
    candidates = select_candidates(data, args)
    write_outputs(candidates, args.output_dir, args)
    print(json.dumps({
        "input_rows": int(len(data)),
        "candidate_rows": int(len(candidates)),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
