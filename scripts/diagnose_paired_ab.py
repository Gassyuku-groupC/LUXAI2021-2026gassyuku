#!/usr/bin/env python3
"""Compare two replay sets on matched map/opponent/seed/player games."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_replays import player_metrics


NAME_PATTERN = re.compile(
    r"map_(?P<map>12|16|24|32)x(?P=map)_vs_(?P<opponent>[A-Za-z0-9_]+)_"
    r"(?P<seed>\d+)_p(?P<player>[01])$"
)


def replay_key(path: Path) -> tuple[int, str, int, int] | None:
    match = NAME_PATTERN.match(path.stem)
    if not match:
        return None
    groups = match.groupdict()
    return (
        int(groups["map"]),
        groups["opponent"],
        int(groups["seed"]),
        int(groups["player"]),
    )


def load_replays(path: Path) -> dict[tuple[int, str, int, int], Path]:
    if not path.exists():
        raise FileNotFoundError(f"Replay path does not exist: {path}")
    files = path.rglob("*.json") if path.is_dir() else [path]
    indexed = {}
    for replay_path in files:
        key = replay_key(replay_path)
        if key is not None:
            indexed[key] = replay_path
    return indexed


def total_city_stats(state: dict[str, Any], player: int) -> dict[str, float]:
    tiles = 0
    fuel = 0.0
    upkeep = 0.0
    for city in state["cities"].values():
        if int(city["team"]) != player:
            continue
        city_tiles = len(city["cityCells"])
        tiles += city_tiles
        fuel += float(city["fuel"])
        upkeep += float(city["lightupkeep"])
    return {
        "tiles": float(tiles),
        "fuel": fuel,
        "upkeep": upkeep,
        "fuel_turns": fuel / max(upkeep, 1.0) if tiles else 0.0,
    }


def closest_turn(states: list[dict[str, Any]], target: int) -> dict[str, Any]:
    return min(states, key=lambda state: abs(int(state["turn"]) - target))


def replay_metrics(path: Path, player: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as replay_file:
        replay = json.load(replay_file)
    metrics = player_metrics(replay, player)
    states = replay["stateful"]
    turn_stats = {
        str(turn): total_city_stats(closest_turn(states, turn), player)
        for turn in (280, 320, 340, 350, 360)
    }
    endgame_states = [state for state in states if int(state["turn"]) >= 320]
    endgame_fuel_turns = [
        total_city_stats(state, player)["fuel_turns"]
        for state in endgame_states
        if total_city_stats(state, player)["tiles"] > 0
    ]
    metrics.update(
        {
            "win": metrics["rank"] == 1,
            "turn_stats": turn_stats,
            "endgame_city_gain_320_350": (
                turn_stats["350"]["tiles"] - turn_stats["320"]["tiles"]
            ),
            "min_endgame_fuel_turns": min(endgame_fuel_turns, default=0.0),
            "fuel_turns_350": turn_stats["350"]["fuel_turns"],
        }
    )
    return metrics


def number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return number(candidate.get(key)) - number(baseline.get(key))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"games": 0}
    return {
        "games": len(rows),
        "candidate_win_rate": mean(row["candidate"]["win"] for row in rows),
        "baseline_win_rate": mean(row["baseline"]["win"] for row in rows),
        "win_rate_delta": mean(row["delta"]["win"] for row in rows),
        "survival_delta": mean(row["delta"]["effective_survival"] for row in rows),
        "city_tiles_delta": mean(row["delta"]["city_tiles"] for row in rows),
        "night_loss_delta": mean(row["delta"]["max_night_city_loss"] for row in rows),
        "worst_candidate_night_loss": max(
            row["candidate"]["max_night_city_loss"] for row in rows
        ),
        "fuel_turns_350_delta": mean(row["delta"]["fuel_turns_350"] for row in rows),
        "endgame_city_gain_delta": mean(
            row["delta"]["endgame_city_gain_320_350"] for row in rows
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Paired A/B Diagnosis",
        "",
        f"- candidate: `{report['candidate_root']}`",
        f"- baseline: `{report['baseline_root']}`",
        f"- common games: `{report['summary']['games']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in report["summary"].items():
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")

    lines.extend(
        [
            "",
            "## Worst Candidate Night Losses",
            "",
            "| map | opponent | seed | player | cand_loss | base_loss | cand_ft350 | base_ft350 | cand_gain320_350 | base_gain320_350 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    worst = sorted(
        report["details"],
        key=lambda row: row["candidate"]["max_night_city_loss"],
        reverse=True,
    )[:20]
    for row in worst:
        map_size, opponent, seed, player = row["key"]
        lines.append(
            f"| {map_size} | {opponent} | {seed} | {player} | "
            f"{row['candidate']['max_night_city_loss']} | "
            f"{row['baseline']['max_night_city_loss']} | "
            f"{row['candidate']['fuel_turns_350']:.2f} | "
            f"{row['baseline']['fuel_turns_350']:.2f} | "
            f"{row['candidate']['endgame_city_gain_320_350']:.1f} | "
            f"{row['baseline']['endgame_city_gain_320_350']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty report instead of failing when no matched replays are found.",
    )
    args = parser.parse_args()

    candidate_replays = load_replays(args.candidate)
    baseline_replays = load_replays(args.baseline)
    common_keys = sorted(set(candidate_replays) & set(baseline_replays))
    if not args.allow_empty:
        if not candidate_replays:
            raise SystemExit(
                "No candidate replays matched the expected name pattern "
                "`map_16x16_vs_OPPONENT_SEED_p0.json`. "
                f"Check --candidate: {args.candidate}"
            )
        if not baseline_replays:
            raise SystemExit(
                "No baseline replays matched the expected name pattern "
                "`map_16x16_vs_OPPONENT_SEED_p0.json`. "
                f"Check --baseline: {args.baseline}"
            )
        if not common_keys:
            raise SystemExit(
                "Candidate and baseline replays were found, but none share the same "
                "map/opponent/seed/player key. Use paired evaluation seeds before "
                "running this comparison."
            )

    details = []
    for key in common_keys:
        map_size, opponent, seed, player = key
        candidate = replay_metrics(candidate_replays[key], player)
        baseline = replay_metrics(baseline_replays[key], player)
        row_delta = {
            "win": number(candidate["win"]) - number(baseline["win"]),
            "effective_survival": delta(candidate, baseline, "effective_survival"),
            "city_tiles": delta(candidate, baseline, "city_tiles"),
            "max_night_city_loss": delta(candidate, baseline, "max_night_city_loss"),
            "fuel_turns_350": delta(candidate, baseline, "fuel_turns_350"),
            "endgame_city_gain_320_350": delta(
                candidate,
                baseline,
                "endgame_city_gain_320_350",
            ),
        }
        details.append(
            {
                "key": [map_size, opponent, seed, player],
                "candidate_file": str(candidate_replays[key]),
                "baseline_file": str(baseline_replays[key]),
                "candidate": candidate,
                "baseline": baseline,
                "delta": row_delta,
            }
        )

    report = {
        "candidate_root": str(args.candidate),
        "baseline_root": str(args.baseline),
        "candidate_games": len(candidate_replays),
        "baseline_games": len(baseline_replays),
        "common_games": len(common_keys),
        "summary": summarize(details),
        "details": details,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, report)
    print(rendered)


if __name__ == "__main__":
    main()
