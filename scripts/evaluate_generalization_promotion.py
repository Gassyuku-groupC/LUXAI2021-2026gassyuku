#!/usr/bin/env python3
"""Evaluate a candidate without allowing maps, opponents, or sides to hide regressions."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from evaluate_replays import player_metrics


NAME_PATTERN = re.compile(
    r"map_(?P<map>12|16|24|32)x(?P=map)_vs_(?P<opponent>[A-Za-z0-9_]+)_"
    r"(?P<seed>\d+)_p(?P<player>[01])$"
)


def mean(items, key):
    return sum(float(item[key]) for item in items) / len(items)


def risk_adjusted_score(summary, args):
    night_excess = max(summary["worst_night_city_loss"] - args.max_night_city_loss, 0)
    side_gap = summary.get("sides", {}).get("city_gap", 0.0)
    side_excess = max(side_gap - args.max_side_city_gap, 0.0)
    score = (
        summary["win_rate"] * args.score_win_weight
        + summary["effective_survival_rate"] * args.score_survival_weight
        + min(summary["mean_city_tiles"] / max(args.score_city_tiles_scale, 1.0), 1.0) * args.score_city_weight
        + max(min(summary.get("mean_city_tile_margin", 0.0) / max(args.score_margin_scale, 1.0), 1.0), -1.0)
        * args.score_margin_weight
        + summary["uranium_rate"] * args.score_uranium_weight
        - night_excess * args.score_night_loss_penalty
        - side_excess * args.score_side_gap_penalty
    )
    return score


def summarize(items):
    if not items:
        return {
            "games": 0,
            "win_rate": 0.0,
            "effective_survival_rate": 0.0,
            "mean_city_tiles": 0.0,
            "mean_city_tile_margin": 0.0,
            "mean_unit_margin": 0.0,
            "uranium_rate": 0.0,
            "worst_night_city_loss": 0,
        }
    return {
        "games": len(items),
        "win_rate": mean(items, "win"),
        "effective_survival_rate": mean(items, "effective_survival"),
        "mean_city_tiles": mean(items, "city_tiles"),
        "mean_city_tile_margin": mean(items, "city_tile_margin"),
        "mean_unit_margin": mean(items, "unit_margin"),
        "uranium_rate": mean(items, "uranium"),
        "worst_night_city_loss": max(item["max_night_city_loss"] for item in items),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-win-rate", type=float, default=0.375)
    parser.add_argument("--min-survival-rate", type=float, default=0.75)
    parser.add_argument("--min-small-map-uranium-rate", type=float, default=0.75)
    parser.add_argument("--max-side-city-gap", type=float, default=0.35)
    parser.add_argument("--max-night-city-loss", type=int, default=10)
    parser.add_argument("--night-loss-mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--catastrophic-night-city-loss", type=int, default=60)
    parser.add_argument("--min-risk-adjusted-score", type=float, default=0.53)
    parser.add_argument("--score-win-weight", type=float, default=0.46)
    parser.add_argument("--score-survival-weight", type=float, default=0.16)
    parser.add_argument("--score-city-weight", type=float, default=0.16)
    parser.add_argument("--score-margin-weight", type=float, default=0.14)
    parser.add_argument("--score-uranium-weight", type=float, default=0.08)
    parser.add_argument("--score-city-tiles-scale", type=float, default=50.0)
    parser.add_argument("--score-margin-scale", type=float, default=50.0)
    parser.add_argument("--score-night-loss-penalty", type=float, default=0.004)
    parser.add_argument("--score-side-gap-penalty", type=float, default=0.12)
    parser.add_argument("--shadow-min-survival-rate", type=float, default=0.50)
    parser.add_argument("--shadow-max-night-city-loss", type=int, default=60)
    parser.add_argument("--shadow-max-side-city-gap", type=float, default=0.80)
    args = parser.parse_args()

    games = []
    for path in args.replays:
        match = NAME_PATTERN.match(path.stem)
        if not match:
            raise ValueError(f"Unexpected replay name: {path.name}")
        with path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
        labels = match.groupdict()
        metrics = player_metrics(replay, int(labels["player"]))
        opponent_metrics = player_metrics(replay, 1 - int(labels["player"]))
        games.append({
            "file": str(path),
            "map_size": int(labels["map"]),
            "opponent": labels["opponent"],
            "seed": int(labels["seed"]),
            "player": int(labels["player"]),
            "win": metrics["rank"] == 1,
            "uranium": metrics["research"] >= 200,
            "opponent_city_tiles": opponent_metrics["city_tiles"],
            "opponent_units": opponent_metrics["units"],
            "city_tile_margin": metrics["city_tiles"] - opponent_metrics["city_tiles"],
            "unit_margin": metrics["units"] - opponent_metrics["units"],
            **metrics,
        })

    grouped = defaultdict(list)
    side_grouped = defaultdict(list)
    for game in games:
        grouped[(game["map_size"], game["opponent"])].append(game)
        side_grouped[(game["map_size"], game["opponent"], game["player"])].append(game)

    checks = []
    group_summaries = {}
    for (map_size, opponent), items in sorted(grouped.items()):
        name = f"{map_size}x{map_size}_vs_{opponent}"
        summary = summarize(items)
        group_summaries[name] = summary
        checks.extend([
            {"name": f"{name}.win_rate", "value": summary["win_rate"],
             "limit": args.min_win_rate, "passed": summary["win_rate"] >= args.min_win_rate},
            {"name": f"{name}.effective_survival", "value": summary["effective_survival_rate"],
             "limit": args.min_survival_rate,
             "passed": summary["effective_survival_rate"] >= args.min_survival_rate},
            {"name": f"{name}.night_loss", "value": summary["worst_night_city_loss"],
             "limit": args.max_night_city_loss,
             "passed": summary["worst_night_city_loss"] <= args.max_night_city_loss},
        ])
        if map_size <= 16:
            checks.append({
                "name": f"{name}.uranium_rate", "value": summary["uranium_rate"],
                "limit": args.min_small_map_uranium_rate,
                "passed": summary["uranium_rate"] >= args.min_small_map_uranium_rate,
            })

        p0 = summarize(side_grouped[(map_size, opponent, 0)])
        p1 = summarize(side_grouped[(map_size, opponent, 1)])
        denominator = max(p0["mean_city_tiles"], p1["mean_city_tiles"], 1.0)
        side_gap = abs(p0["mean_city_tiles"] - p1["mean_city_tiles"]) / denominator
        group_summaries[name]["sides"] = {"p0": p0, "p1": p1, "city_gap": side_gap}
        group_summaries[name]["risk_adjusted_score"] = risk_adjusted_score(group_summaries[name], args)
        group_summaries[name]["night_loss_soft_penalty"] = (
            max(summary["worst_night_city_loss"] - args.max_night_city_loss, 0)
            * args.score_night_loss_penalty
        )
        checks.append({
            "name": f"{name}.side_city_gap", "value": side_gap,
            "limit": args.max_side_city_gap, "passed": side_gap <= args.max_side_city_gap,
        })
        checks.append({
            "name": f"{name}.risk_adjusted_score",
            "value": group_summaries[name]["risk_adjusted_score"],
            "limit": args.min_risk_adjusted_score,
            "passed": group_summaries[name]["risk_adjusted_score"] >= args.min_risk_adjusted_score,
        })
        if args.night_loss_mode == "soft":
            for check in checks:
                if check["name"] == f"{name}.night_loss":
                    check["soft"] = True
                    check["passed"] = (
                        summary["worst_night_city_loss"] <= args.max_night_city_loss
                        or (
                            summary["worst_night_city_loss"] <= args.catastrophic_night_city_loss
                            and group_summaries[name]["risk_adjusted_score"] >= args.min_risk_adjusted_score
                        )
                    )
                    check["catastrophic_limit"] = args.catastrophic_night_city_loss
                    break

    shadow_failed_checks = []
    excessive_side_gaps = 0
    for name, summary in group_summaries.items():
        if summary["effective_survival_rate"] < args.shadow_min_survival_rate:
            shadow_failed_checks.append({
                "name": f"{name}.shadow_survival",
                "value": summary["effective_survival_rate"],
                "limit": args.shadow_min_survival_rate,
            })
        if summary["worst_night_city_loss"] > args.shadow_max_night_city_loss:
            shadow_failed_checks.append({
                "name": f"{name}.shadow_night_loss",
                "value": summary["worst_night_city_loss"],
                "limit": args.shadow_max_night_city_loss,
            })
        if summary["sides"]["city_gap"] > args.shadow_max_side_city_gap:
            excessive_side_gaps += 1

    # A single noisy matchup does not reset accumulated learning. A majority
    # of severely side-biased groups is treated as catastrophic degradation.
    if excessive_side_gaps > len(group_summaries) / 2:
        shadow_failed_checks.append({
            "name": "shadow.excessive_side_gap_groups",
            "value": excessive_side_gaps,
            "limit": len(group_summaries) // 2,
        })

    result = {
        "promote": all(check["passed"] for check in checks),
        "shadow_safe": not shadow_failed_checks,
        "criteria": vars(args) | {"replays": None, "output": None},
        "groups": group_summaries,
        "failed_checks": [check for check in checks if not check["passed"]],
        "shadow_failed_checks": shadow_failed_checks,
        "checks": checks,
        "details": games,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
