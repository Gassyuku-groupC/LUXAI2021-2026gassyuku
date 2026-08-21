#!/usr/bin/env python3
"""Score offline Lux replays with a trained suggestion reward scorer."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_suggestion_labels import FIELDNAMES, make_label  # noqa: E402
from extract_strategy_features import iter_replay_paths, load_replay_bundle  # noqa: E402
from mine_adjacent_fuel_support import mine_bundle  # noqa: E402


def infer_eval_side(path: str) -> int | None:
    match = re.search(r"_p([01])(?:\.json)?$", Path(path).name)
    return int(match.group(1)) if match else None


def city_tiles_by_turn(states: list[dict], team: int) -> list[tuple[int, int]]:
    rows = []
    for state in states:
        turn = int(state.get("turn", len(rows)))
        tiles = sum(
            len(city.get("cityCells") or [])
            for city in (state.get("cities") or {}).values()
            if int(city.get("team", -1)) == team
        )
        rows.append((turn, tiles))
    return rows


def night_losses(states: list[dict], team: int) -> list[dict]:
    rows = []
    tiles = city_tiles_by_turn(states, team)
    for (_, previous), (turn, current) in zip(tiles, tiles[1:]):
        if current < previous and turn % 40 >= 30:
            rows.append({"turn": turn, "lost": previous - current, "before": previous, "after": current})
    return rows


def night_cycle(turn: int) -> int:
    return int(turn) // 40


def replay_team_loss_summary(patterns: list[str]) -> pd.DataFrame:
    rows = []
    for path in iter_replay_paths(patterns):
        bundle = load_replay_bundle(path)
        if not bundle:
            continue
        side = infer_eval_side(str(path))
        teams = [side] if side is not None else [0, 1]
        for team in teams:
            losses = night_losses(bundle["states"], team)
            worst = max(losses, key=lambda row: row["lost"], default=None)
            rows.append(
                {
                    "source_file": str(path),
                    "team": team,
                    "worst_night_loss": int(worst["lost"]) if worst else 0,
                    "worst_night_loss_turn": int(worst["turn"]) if worst else -1,
                    "night_loss_events": len(losses),
                    "total_night_loss": int(sum(row["lost"] for row in losses)),
                }
            )
    return pd.DataFrame(rows)


def replay_loss_events(patterns: list[str]) -> pd.DataFrame:
    rows = []
    for path in iter_replay_paths(patterns):
        bundle = load_replay_bundle(path)
        if not bundle:
            continue
        side = infer_eval_side(str(path))
        teams = [side] if side is not None else [0, 1]
        for team in teams:
            for loss in night_losses(bundle["states"], team):
                rows.append(
                    {
                        "source_file": str(path),
                        "team": int(team),
                        "loss_turn": int(loss["turn"]),
                        "night_cycle": night_cycle(int(loss["turn"])),
                        "lost": int(loss["lost"]),
                        "before": int(loss["before"]),
                        "after": int(loss["after"]),
                    }
                )
    return pd.DataFrame(rows)


def build_labels_from_replays(patterns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    labels = []
    for replay_path in iter_replay_paths(patterns):
        bundle = load_replay_bundle(replay_path)
        if not bundle:
            continue
        for event in mine_bundle(bundle, args):
            label = make_label(event, args)
            if label:
                labels.append(label)
    return pd.DataFrame(labels, columns=FIELDNAMES)


def load_labels(args: argparse.Namespace) -> pd.DataFrame:
    if args.labels:
        data = pd.read_csv(args.labels)
    else:
        data = build_labels_from_replays(args.patterns, args)
    if data.empty:
        return data
    if args.map_size:
        data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == args.map_size]
    if args.max_turn:
        data = data[pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int) <= args.max_turn]
    return data.reset_index(drop=True)


def prepare_features(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = 0.0
    return frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def score_labels(data: pd.DataFrame, model_path: Path) -> tuple[pd.DataFrame, dict]:
    payload = joblib.load(model_path)
    features = payload["features"]
    penalty_model = payload["penalty_model"]
    reward_model = payload["reward_model"]
    x = prepare_features(data, features)
    scored = data.copy()
    scored["pred_penalty_prob"] = penalty_model.predict_proba(x)[:, 1]
    scored["pred_reward_value"] = reward_model.predict(x)
    scored["pred_risk_score"] = scored["pred_penalty_prob"] * (-scored["pred_reward_value"]).clip(lower=0)
    scored.sort_values(["source_file", "team", "pred_risk_score"], ascending=[True, True, False], inplace=True)
    return scored, payload


def summarize_hits(
    scored: pd.DataFrame,
    loss_summary: pd.DataFrame,
    top_k: int,
    lookahead_turns: int,
) -> dict:
    if scored.empty:
        return {"rows": 0}
    rows = []
    grouped = scored.groupby(["source_file", "team"], dropna=False)
    for (source_file, team), group in grouped:
        top = group.head(top_k).copy()
        loss_row = loss_summary[
            (loss_summary["source_file"].astype(str) == str(source_file))
            & (loss_summary["team"].astype(int) == int(team))
        ]
        if loss_row.empty:
            worst_loss = 0
            worst_turn = -1
        else:
            item = loss_row.iloc[0]
            worst_loss = int(item["worst_night_loss"])
            worst_turn = int(item["worst_night_loss_turn"])
        top_turns = [int(float(value)) for value in top["turn"].tolist()]
        hit = any(0 <= worst_turn - turn <= lookahead_turns for turn in top_turns)
        first_warning = min(top_turns) if top_turns else -1
        rows.append(
            {
                "source_file": source_file,
                "team": int(team),
                "suggestions": int(len(group)),
                "top_k": int(len(top)),
                "top_turns": ";".join(str(turn) for turn in top_turns),
                "first_top_turn": first_warning,
                "worst_night_loss": worst_loss,
                "worst_night_loss_turn": worst_turn,
                "hit_worst_loss_window": int(hit),
                "lead_time_to_worst": worst_turn - first_warning if worst_turn >= 0 and first_warning >= 0 else "",
                "max_pred_penalty_prob": float(top["pred_penalty_prob"].max()) if len(top) else 0.0,
                "max_pred_risk_score": float(top["pred_risk_score"].max()) if len(top) else 0.0,
            }
        )
    report = pd.DataFrame(rows)
    severe = report[report["worst_night_loss"] >= 10]
    return {
        "groups": int(len(report)),
        "suggestion_rows": int(len(scored)),
        "top_k": int(top_k),
        "lookahead_turns": int(lookahead_turns),
        "hit_rate_all": float(report["hit_worst_loss_window"].mean()) if len(report) else 0.0,
        "severe_groups": int(len(severe)),
        "hit_rate_severe": float(severe["hit_worst_loss_window"].mean()) if len(severe) else 0.0,
        "mean_lead_time_severe": float(
            pd.to_numeric(severe["lead_time_to_worst"], errors="coerce").dropna().mean()
        ) if len(severe) else 0.0,
        "worst_groups": report.sort_values("worst_night_loss", ascending=False).head(20).to_dict(orient="records"),
    }


def top_per_night(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    data = scored.copy()
    data["night_cycle"] = pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int).map(night_cycle)
    data.sort_values(["source_file", "team", "night_cycle", "pred_risk_score"], ascending=[True, True, True, False], inplace=True)
    return data.groupby(["source_file", "team", "night_cycle"], dropna=False).head(1).reset_index(drop=True)


def top_per_city(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    data = scored.copy()
    data.sort_values(["source_file", "team", "suggested_city_id", "pred_risk_score"], ascending=[True, True, True, False], inplace=True)
    return data.groupby(["source_file", "team", "suggested_city_id"], dropna=False).head(1).reset_index(drop=True)


def top_per_night_city(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    data = scored.copy()
    data["night_cycle"] = pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int).map(night_cycle)
    data.sort_values(
        ["source_file", "team", "night_cycle", "suggested_city_id", "pred_risk_score"],
        ascending=[True, True, True, True, False],
        inplace=True,
    )
    return data.groupby(["source_file", "team", "night_cycle", "suggested_city_id"], dropna=False).head(1).reset_index(drop=True)


def hit_any_big_loss_report(
    scored: pd.DataFrame,
    loss_events: pd.DataFrame,
    lookahead_turns: int,
    big_loss_threshold: int,
) -> tuple[pd.DataFrame, dict]:
    if loss_events.empty:
        return pd.DataFrame(), {"big_loss_events": 0}
    big = loss_events[pd.to_numeric(loss_events["lost"], errors="coerce").fillna(0).astype(int) >= big_loss_threshold].copy()
    rows = []
    for _, event in big.iterrows():
        source_file = str(event["source_file"])
        team = int(event["team"])
        loss_turn = int(event["loss_turn"])
        candidates = scored[
            (scored["source_file"].astype(str) == source_file)
            & (pd.to_numeric(scored["team"], errors="coerce").fillna(-1).astype(int) == team)
        ].copy()
        if candidates.empty:
            pre = candidates
        else:
            turns = pd.to_numeric(candidates["turn"], errors="coerce").fillna(-9999).astype(int)
            pre = candidates[(turns <= loss_turn) & (turns >= loss_turn - lookahead_turns)]
        hit = not pre.empty
        top = pre.sort_values("pred_risk_score", ascending=False).head(1)
        rows.append(
            {
                "source_file": source_file,
                "team": team,
                "loss_turn": loss_turn,
                "night_cycle": int(event["night_cycle"]),
                "lost": int(event["lost"]),
                "hit_any_big_loss": int(hit),
                "best_warning_turn": int(float(top.iloc[0]["turn"])) if hit else "",
                "lead_time": loss_turn - int(float(top.iloc[0]["turn"])) if hit else "",
                "best_warning_city_id": str(top.iloc[0].get("suggested_city_id", "")) if hit else "",
                "best_pred_risk_score": float(top.iloc[0]["pred_risk_score"]) if hit else 0.0,
                "best_pred_penalty_prob": float(top.iloc[0]["pred_penalty_prob"]) if hit else 0.0,
            }
        )
    report = pd.DataFrame(rows)
    summary = {
        "big_loss_threshold": int(big_loss_threshold),
        "lookahead_turns": int(lookahead_turns),
        "big_loss_events": int(len(report)),
        "hit_any_big_loss_rate": float(report["hit_any_big_loss"].mean()) if len(report) else 0.0,
        "mean_lead_time": float(pd.to_numeric(report["lead_time"], errors="coerce").dropna().mean()) if len(report) else 0.0,
    }
    return report, summary


def dedup_summary(
    name: str,
    scored: pd.DataFrame,
    loss_events: pd.DataFrame,
    lookahead_turns: int,
    big_loss_threshold: int,
) -> tuple[pd.DataFrame, dict]:
    report, summary = hit_any_big_loss_report(scored, loss_events, lookahead_turns, big_loss_threshold)
    summary["dedup_mode"] = name
    summary["warning_rows"] = int(len(scored))
    return report, summary


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score suggestion labels or replay files.")
    parser.add_argument("patterns", nargs="*", help="Replay glob patterns when --labels is not provided.")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/diagnostic_layer/suggestion_reward_lgbm_v1b_from_best/suggestion_reward_lgbm.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/suggestion_reward_lgbm_v1b_from_best/offline_score"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--city-fuel-turns-lt", type=float, default=2.5)
    parser.add_argument("--min-cargo-fuel", type=float, default=80.0)
    parser.add_argument("--turns-to-night-lte", type=int, default=3)
    parser.add_argument("--max-turn", type=int, default=240)
    parser.add_argument("--include-night", action="store_true", default=True)
    parser.add_argument("--exclude-night", dest="include_night", action="store_false")
    parser.add_argument("--include-supporting", action="store_true", default=True)
    parser.add_argument("--exclude-supporting", dest="include_supporting", action="store_false")
    parser.add_argument("--all-teams", action="store_true")
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--label-version", default="fuel_support_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lookahead-turns", type=int, default=20)
    parser.add_argument("--big-loss-threshold", type=int, default=10)
    args = parser.parse_args()

    if not args.labels and not args.patterns:
        raise ValueError("Pass --labels or at least one replay glob pattern.")

    labels = load_labels(args)
    scored, _ = score_labels(labels, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / "suggestion_scores.csv"
    write_csv(scored_path, scored)

    summary = {
        "model": str(args.model),
        "labels": str(args.labels) if args.labels else None,
        "patterns": args.patterns,
        "rows": int(len(scored)),
        "scored": str(scored_path),
    }
    if not args.labels:
        loss_summary = replay_team_loss_summary(args.patterns)
        loss_events = replay_loss_events(args.patterns)
        hit_summary = summarize_hits(scored, loss_summary, args.top_k, args.lookahead_turns)
        hit_report = []
        if not scored.empty:
            grouped = scored.groupby(["source_file", "team"], dropna=False)
            for (source_file, team), group in grouped:
                top = group.head(args.top_k).copy()
                loss_row = loss_summary[
                    (loss_summary["source_file"].astype(str) == str(source_file))
                    & (loss_summary["team"].astype(int) == int(team))
                ]
                worst_loss = int(loss_row.iloc[0]["worst_night_loss"]) if not loss_row.empty else 0
                worst_turn = int(loss_row.iloc[0]["worst_night_loss_turn"]) if not loss_row.empty else -1
                top_turns = [int(float(value)) for value in top["turn"].tolist()]
                hit_report.append(
                    {
                        "source_file": source_file,
                        "team": int(team),
                        "top_turns": ";".join(str(turn) for turn in top_turns),
                        "worst_night_loss": worst_loss,
                        "worst_night_loss_turn": worst_turn,
                        "hit_worst_loss_window": int(any(0 <= worst_turn - turn <= args.lookahead_turns for turn in top_turns)),
                        "max_pred_penalty_prob": float(top["pred_penalty_prob"].max()) if len(top) else 0.0,
                        "max_pred_risk_score": float(top["pred_risk_score"].max()) if len(top) else 0.0,
                    }
                )
        hit_report_path = args.output_dir / "worst_loss_hit_report.csv"
        write_csv(hit_report_path, pd.DataFrame(hit_report))
        summary["worst_loss_validation"] = hit_summary
        summary["worst_loss_hit_report"] = str(hit_report_path)

        any_big_report, any_big_summary = dedup_summary(
            "raw_all_suggestions",
            scored,
            loss_events,
            args.lookahead_turns,
            args.big_loss_threshold,
        )
        per_night = top_per_night(scored)
        per_night_report, per_night_summary = dedup_summary(
            "per_night_top1",
            per_night,
            loss_events,
            args.lookahead_turns,
            args.big_loss_threshold,
        )
        per_city = top_per_city(scored)
        per_city_report, per_city_summary = dedup_summary(
            "per_city_top1",
            per_city,
            loss_events,
            args.lookahead_turns,
            args.big_loss_threshold,
        )
        per_night_city = top_per_night_city(scored)
        per_night_city_report, per_night_city_summary = dedup_summary(
            "per_night_city_top1",
            per_night_city,
            loss_events,
            args.lookahead_turns,
            args.big_loss_threshold,
        )
        any_big_path = args.output_dir / "big_loss_event_hit_report.csv"
        per_night_path = args.output_dir / "per_night_top1_hit_report.csv"
        per_city_path = args.output_dir / "per_city_top1_hit_report.csv"
        per_night_city_path = args.output_dir / "per_night_city_top1_hit_report.csv"
        write_csv(any_big_path, any_big_report)
        write_csv(per_night_path, per_night_report)
        write_csv(per_city_path, per_city_report)
        write_csv(per_night_city_path, per_night_city_report)
        write_csv(args.output_dir / "per_night_top1_scores.csv", per_night)
        write_csv(args.output_dir / "per_city_top1_scores.csv", per_city)
        write_csv(args.output_dir / "per_night_city_top1_scores.csv", per_night_city)
        summary["any_big_loss_validation"] = {
            "raw_all_suggestions": any_big_summary,
            "per_night_top1": per_night_summary,
            "per_city_top1": per_city_summary,
            "per_night_city_top1": per_night_city_summary,
        }
        summary["big_loss_event_hit_report"] = str(any_big_path)
        summary["per_night_top1_hit_report"] = str(per_night_path)
        summary["per_city_top1_hit_report"] = str(per_city_path)
        summary["per_night_city_top1_hit_report"] = str(per_night_city_path)

    summary_path = args.output_dir / "score_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
