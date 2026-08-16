#!/usr/bin/env python3
"""Validate offline candidate-action suggestions.

Focuses on missed_safe_bcity_window: whether these suggestions appear in states
with healthy buffers, low future loss, and poor final margin where extra scale
could plausibly matter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "rank",
    "turn",
    "cycle_turn",
    "turns_to_night",
    "city_tiles",
    "workers",
    "research",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "low_fuel_city_lt5",
    "future_team_loss_20",
    "final_city_tile_margin",
    "bad_score_delta",
    "actual_big_risk",
    "best_expand_safe_expansion",
    "best_expand_big_risk",
    "bcity_big_risk",
    "bcity_safe_expansion",
    "bcity_success",
    "no_expand_big_risk",
    "no_expand_success",
]


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    for column in ["suggestion", "source_opponent", "phase", "file", "actual_action"]:
        if column in data:
            data[column] = data[column].fillna("").astype(str)
    return data


def turn_bucket(turn: float) -> str:
    turn = int(turn)
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


def summarize(group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "rows": 0,
            "episodes": 0,
            "loss_rate": 0.0,
            "mean_turn": 0.0,
            "mean_margin": 0.0,
            "mean_loss20": 0.0,
            "mean_min_fuel": 0.0,
            "mean_p25_fuel": 0.0,
            "mean_bcity_safe": 0.0,
            "mean_bcity_big_risk": 0.0,
        }
    return {
        "rows": int(len(group)),
        "episodes": int(group["file"].nunique()) if "file" in group else 0,
        "loss_rate": float((group["rank"] == 2).mean()) if "rank" in group else 0.0,
        "mean_turn": float(group["turn"].mean()),
        "mean_margin": float(group["final_city_tile_margin"].mean()),
        "mean_loss20": float(group["future_team_loss_20"].mean()),
        "mean_min_fuel": float(group["min_city_fuel_turns"].mean()),
        "mean_p25_fuel": float(group["p25_city_fuel_turns"].mean()),
        "mean_bcity_safe": float(group["bcity_safe_expansion"].mean()),
        "mean_bcity_big_risk": float(group["bcity_big_risk"].mean()),
        "mean_no_expand_big_risk": float(group["no_expand_big_risk"].mean()),
        "mean_bcity_success": float(group["bcity_success"].mean()),
        "mean_no_expand_success": float(group["no_expand_success"].mean()),
    }


def grouped_summary(data: pd.DataFrame, key: str) -> dict:
    if key not in data:
        return {}
    return {str(value): summarize(group) for value, group in data.groupby(key, dropna=False)}


def episode_level(data: pd.DataFrame) -> pd.DataFrame:
    missed = data[data["suggestion"] == "missed_safe_bcity_window"].copy()
    rows = []
    for file_value, group in data.groupby("file", dropna=False):
        missed_group = missed[missed["file"] == file_value]
        first = group.iloc[0]
        rows.append(
            {
                "file": file_value,
                "source_opponent": first.get("source_opponent", ""),
                "rank": int(float(first.get("rank", 0) or 0)),
                "final_city_tile_margin": float(first.get("final_city_tile_margin", 0) or 0),
                "states": int(len(group)),
                "missed_safe_bcity_windows": int(len(missed_group)),
                "missed_rate": float(len(missed_group) / max(len(group), 1)),
                "mean_missed_turn": float(missed_group["turn"].mean()) if len(missed_group) else 0.0,
                "mean_missed_min_fuel": float(missed_group["min_city_fuel_turns"].mean()) if len(missed_group) else 0.0,
                "mean_missed_p25_fuel": float(missed_group["p25_city_fuel_turns"].mean()) if len(missed_group) else 0.0,
                "mean_missed_bcity_safe": float(missed_group["bcity_safe_expansion"].mean()) if len(missed_group) else 0.0,
                "mean_missed_bcity_big_risk": float(missed_group["bcity_big_risk"].mean()) if len(missed_group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def validation_verdict(data: pd.DataFrame, episodes: pd.DataFrame, args: argparse.Namespace) -> dict:
    missed = data[data["suggestion"] == "missed_safe_bcity_window"]
    notes = []
    useful = False
    if missed.empty:
        notes.append("No missed_safe_bcity_window suggestions found.")
        return {"useful_for_next_step": False, "notes": notes}

    healthy_rate = (
        (missed["p25_city_fuel_turns"] >= args.min_p25_fuel)
        & (missed["bcity_big_risk"] <= args.max_bcity_big_risk)
        & (missed["future_team_loss_20"] <= args.max_future_loss20)
    ).mean()
    loss_episodes = episodes[episodes["rank"] == 2]
    loss_with_missed = (loss_episodes["missed_safe_bcity_windows"] > 0).mean() if len(loss_episodes) else 0.0
    loss_missed_rate = loss_episodes["missed_rate"].mean() if len(loss_episodes) else 0.0

    if healthy_rate >= args.min_healthy_rate:
        notes.append(f"Most missed windows look healthy enough for offline expansion-label use: healthy_rate={healthy_rate:.3f}.")
        useful = True
    else:
        notes.append(f"Missed windows are noisy: healthy_rate={healthy_rate:.3f}.")

    if loss_with_missed >= 0.5:
        notes.append(f"Missed windows cover many losing episodes: loss_with_missed={loss_with_missed:.3f}.")
    else:
        notes.append(f"Missed windows have limited losing-episode coverage: loss_with_missed={loss_with_missed:.3f}.")

    notes.append(f"Average missed rate in losing episodes: {loss_missed_rate:.3f}.")
    return {
        "useful_for_next_step": bool(useful),
        "healthy_rate": float(healthy_rate),
        "loss_with_missed_rate": float(loss_with_missed),
        "loss_missed_rate": float(loss_missed_rate),
        "notes": notes,
    }


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Candidate Action Suggestion Validation",
        "",
        f"- input: `{summary['input_csv']}`",
        f"- rows: {summary['rows']}",
        f"- missed_safe_bcity_window rows: {summary['missed_rows']}",
        f"- useful_for_next_step: `{summary['verdict']['useful_for_next_step']}`",
    ]
    for note in summary["verdict"]["notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Overall", ""])
    for key, value in summary["overall"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## By Rank", ""])
    lines.append("| rank | rows | episodes | loss_rate | margin | loss20 | min fuel | p25 fuel | bcity safe | bcity big risk |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, item in summary["missed_by_rank"].items():
        lines.append(
            f"| {key} | {item['rows']} | {item['episodes']} | {item['loss_rate']:.3f} | "
            f"{item['mean_margin']:.3f} | {item['mean_loss20']:.3f} | "
            f"{item['mean_min_fuel']:.3f} | {item['mean_p25_fuel']:.3f} | "
            f"{item['mean_bcity_safe']:.3f} | {item['mean_bcity_big_risk']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate candidate-action suggestions.")
    parser.add_argument("--input", type=Path, default=Path("outputs/diagnostic_layer/best_agent_candidate_action_suggestions_v2_16/candidate_action_suggestions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/best_agent_candidate_action_suggestions_v2_16/validation"))
    parser.add_argument("--min-p25-fuel", type=float, default=10.0)
    parser.add_argument("--max-bcity-big-risk", type=float, default=0.12)
    parser.add_argument("--max-future-loss20", type=float, default=1.0)
    parser.add_argument("--min-healthy-rate", type=float, default=0.70)
    args = parser.parse_args()

    data = load_data(args.input)
    data["turn_bucket"] = data["turn"].map(turn_bucket)
    missed = data[data["suggestion"] == "missed_safe_bcity_window"].copy()
    episodes = episode_level(data)

    summary = {
        "input_csv": str(args.input),
        "rows": int(len(data)),
        "missed_rows": int(len(missed)),
        "overall": summarize(missed),
        "missed_by_source_opponent": grouped_summary(missed, "source_opponent"),
        "missed_by_rank": grouped_summary(missed, "rank"),
        "missed_by_turn_bucket": grouped_summary(missed, "turn_bucket"),
        "missed_by_phase": grouped_summary(missed, "phase"),
        "episode_summary": {
            "episodes": int(len(episodes)),
            "episodes_with_missed": int((episodes["missed_safe_bcity_windows"] > 0).sum()),
            "mean_missed_per_episode": float(episodes["missed_safe_bcity_windows"].mean()),
            "by_rank": {
                str(rank): {
                    "episodes": int(len(group)),
                    "episodes_with_missed": int((group["missed_safe_bcity_windows"] > 0).sum()),
                    "mean_missed": float(group["missed_safe_bcity_windows"].mean()),
                    "mean_margin": float(group["final_city_tile_margin"].mean()),
                }
                for rank, group in episodes.groupby("rank")
            },
        },
    }
    summary["verdict"] = validation_verdict(data, episodes, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output_dir / "episode_missed_safe_bcity_summary.csv", index=False, encoding="utf-8")
    missed.to_csv(args.output_dir / "missed_safe_bcity_windows.csv", index=False, encoding="utf-8")
    (args.output_dir / "missed_safe_bcity_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "missed_safe_bcity_validation.md", summary)
    print(json.dumps(summary["verdict"], indent=2, ensure_ascii=False))
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
