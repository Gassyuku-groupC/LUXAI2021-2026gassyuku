#!/usr/bin/env python3
"""Build a combined diagnostic report from late warning and suggestion scorers."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_late_big_loss_labels import add_trends, make_label as make_late_label  # noqa: E402
from extract_strategy_features import iter_replay_paths, load_replay_bundle, rows_for_bundle  # noqa: E402
from score_suggestion_reward import (  # noqa: E402
    build_labels_from_replays,
    dedup_summary,
    hit_any_big_loss_report,
    replay_loss_events,
    score_labels,
    top_per_night,
    write_csv,
)


def prepare_features(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = 0.0
    return frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def build_late_labels_from_replays(patterns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    labels = []
    horizons = sorted({5, 10, args.late_horizon, 30})
    for replay_path in iter_replay_paths(patterns):
        bundle = load_replay_bundle(replay_path)
        if not bundle:
            continue
        if args.map_size and int(bundle["width"]) != args.map_size:
            continue
        rows = rows_for_bundle(bundle, horizons)
        rows = add_trends(rows, args.trend_window)
        for row in rows:
            label = make_late_label(row, args)
            if label:
                labels.append(label)
    return pd.DataFrame(labels)


def score_late_labels(data: pd.DataFrame, model_path: Path) -> tuple[pd.DataFrame, dict]:
    payload = joblib.load(model_path)
    model = payload["model"]
    features = payload["features"]
    x = prepare_features(data, features)
    scored = data.copy()
    scored["source_file"] = scored["file"].astype(str)
    scored["pred_late_big_loss_prob"] = model.predict_proba(x)[:, 1]
    scored["pred_late_big_loss_warning"] = (
        scored["pred_late_big_loss_prob"] >= float(payload.get("threshold", 0.35))
    ).astype(int)
    scored.sort_values(["source_file", "team", "pred_late_big_loss_prob"], ascending=[True, True, False], inplace=True)
    return scored, payload


def top_late_per_night(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    data = scored.copy()
    data["night_cycle"] = pd.to_numeric(data["turn"], errors="coerce").fillna(0).astype(int) // 40
    data.sort_values(
        ["source_file", "team", "night_cycle", "pred_late_big_loss_prob"],
        ascending=[True, True, True, False],
        inplace=True,
    )
    return data.groupby(["source_file", "team", "night_cycle"], dropna=False).head(1).reset_index(drop=True)


def late_hit_report(scored: pd.DataFrame, loss_events: pd.DataFrame, lookahead_turns: int, big_loss_threshold: int) -> tuple[pd.DataFrame, dict]:
    if scored.empty:
        return pd.DataFrame(), {"big_loss_events": 0, "hit_any_big_loss_rate": 0.0}
    renamed = scored.copy()
    renamed["pred_risk_score"] = renamed["pred_late_big_loss_prob"]
    renamed["pred_penalty_prob"] = renamed["pred_late_big_loss_prob"]
    renamed["suggested_city_id"] = ""
    return hit_any_big_loss_report(renamed, loss_events, lookahead_turns, big_loss_threshold)


def overlap_report(late_scored: pd.DataFrame, suggestion_scored: pd.DataFrame, loss_events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if loss_events.empty:
        return pd.DataFrame()
    big = loss_events[pd.to_numeric(loss_events["lost"], errors="coerce").fillna(0).astype(int) >= args.big_loss_threshold]
    rows = []
    for _, event in big.iterrows():
        source_file = str(event["source_file"])
        team = int(event["team"])
        loss_turn = int(event["loss_turn"])
        late_group = late_scored[
            (late_scored["source_file"].astype(str) == source_file)
            & (pd.to_numeric(late_scored["team"], errors="coerce").fillna(-1).astype(int) == team)
        ]
        suggestion_group = suggestion_scored[
            (suggestion_scored["source_file"].astype(str) == source_file)
            & (pd.to_numeric(suggestion_scored["team"], errors="coerce").fillna(-1).astype(int) == team)
        ]
        late_turns = pd.to_numeric(late_group["turn"], errors="coerce").fillna(-9999).astype(int) if not late_group.empty else pd.Series(dtype=int)
        sug_turns = pd.to_numeric(suggestion_group["turn"], errors="coerce").fillna(-9999).astype(int) if not suggestion_group.empty else pd.Series(dtype=int)
        late_pre = late_group[(late_turns <= loss_turn) & (late_turns >= loss_turn - args.lookahead_turns)] if not late_group.empty else late_group
        sug_pre = suggestion_group[(sug_turns <= loss_turn) & (sug_turns >= loss_turn - args.lookahead_turns)] if not suggestion_group.empty else suggestion_group
        late_hit = not late_pre.empty
        sug_hit = not sug_pre.empty
        late_top = late_pre.sort_values("pred_late_big_loss_prob", ascending=False).head(1)
        sug_top = sug_pre.sort_values("pred_risk_score", ascending=False).head(1)
        rows.append(
            {
                "source_file": source_file,
                "team": team,
                "loss_turn": loss_turn,
                "night_cycle": int(event["night_cycle"]),
                "lost": int(event["lost"]),
                "late_hit": int(late_hit),
                "suggestion_hit": int(sug_hit),
                "combined_hit": int(late_hit or sug_hit),
                "both_hit": int(late_hit and sug_hit),
                "late_best_turn": int(float(late_top.iloc[0]["turn"])) if late_hit else "",
                "late_lead_time": loss_turn - int(float(late_top.iloc[0]["turn"])) if late_hit else "",
                "late_prob": float(late_top.iloc[0]["pred_late_big_loss_prob"]) if late_hit else 0.0,
                "suggestion_best_turn": int(float(sug_top.iloc[0]["turn"])) if sug_hit else "",
                "suggestion_lead_time": loss_turn - int(float(sug_top.iloc[0]["turn"])) if sug_hit else "",
                "suggestion_risk_score": float(sug_top.iloc[0]["pred_risk_score"]) if sug_hit else 0.0,
            }
        )
    return pd.DataFrame(rows)


def hit_summary(report: pd.DataFrame, prefix: str) -> dict:
    if report.empty:
        return {f"{prefix}_events": 0}
    return {
        f"{prefix}_events": int(len(report)),
        f"{prefix}_hit_rate": float(report[f"{prefix}_hit"].mean()) if f"{prefix}_hit" in report else 0.0,
    }


def thresholded_overlap_summary(overlap: pd.DataFrame, late_threshold: float, suggestion_threshold: float) -> dict:
    if overlap.empty:
        return {
            "big_loss_events": 0,
            "late_threshold": float(late_threshold),
            "suggestion_risk_threshold": float(suggestion_threshold),
            "actionable_combined_hit_rate": 0.0,
        }
    late_actionable = (overlap["late_hit"] == 1) & (
        pd.to_numeric(overlap["late_prob"], errors="coerce").fillna(0.0) >= late_threshold
    )
    suggestion_actionable = (overlap["suggestion_hit"] == 1) & (
        pd.to_numeric(overlap["suggestion_risk_score"], errors="coerce").fillna(0.0) >= suggestion_threshold
    )
    combined = late_actionable | suggestion_actionable
    return {
        "big_loss_events": int(len(overlap)),
        "late_threshold": float(late_threshold),
        "suggestion_risk_threshold": float(suggestion_threshold),
        "actionable_combined_hit_rate": float(combined.mean()),
        "late_actionable_hit_rate": float(late_actionable.mean()),
        "suggestion_actionable_hit_rate": float(suggestion_actionable.mean()),
        "late_only_actionable_hits": int((late_actionable & ~suggestion_actionable).sum()),
        "suggestion_only_actionable_hits": int((~late_actionable & suggestion_actionable).sum()),
        "both_actionable_hits": int((late_actionable & suggestion_actionable).sum()),
        "actionable_misses": int((~combined).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score replay files with combined late/suggestion risk scorers.")
    parser.add_argument("patterns", nargs="+", help="Replay glob patterns.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/combined_risk_report_v1"))
    parser.add_argument("--suggestion-model", type=Path, default=Path("outputs/diagnostic_layer/suggestion_reward_lgbm_v1b_from_best/suggestion_reward_lgbm.joblib"))
    parser.add_argument("--late-model", type=Path, default=Path("outputs/diagnostic_layer/late_big_loss_warning_lgbm_v1_public_v2_16/late_big_loss_warning_lgbm.joblib"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--start-turn", type=int, default=160)
    parser.add_argument("--late-max-turn", dest="max_turn", type=int, default=360)
    parser.add_argument("--suggestion-max-turn", type=int, default=240)
    parser.add_argument("--late-horizon", type=int, default=20, choices=[5, 10, 20, 30])
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 10, 20, 30], help=argparse.SUPPRESS)
    parser.add_argument("--big-loss-threshold", type=int, default=10)
    parser.add_argument("--trend-window", type=int, default=10)
    parser.add_argument("--label-version", default="late_big_loss_warning_v1")
    parser.add_argument("--city-fuel-turns-lt", type=float, default=2.5)
    parser.add_argument("--min-cargo-fuel", type=float, default=80.0)
    parser.add_argument("--turns-to-night-lte", type=int, default=3)
    parser.add_argument("--include-night", action="store_true", default=True)
    parser.add_argument("--include-supporting", action="store_true", default=True)
    parser.add_argument("--all-teams", action="store_true")
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--lookahead-turns", type=int, default=20)
    parser.add_argument("--late-action-threshold", type=float, default=0.35)
    parser.add_argument("--suggestion-action-threshold", type=float, default=2.0)
    args = parser.parse_args()
    args.horizon = args.late_horizon

    args.output_dir.mkdir(parents=True, exist_ok=True)
    late_labels = build_late_labels_from_replays(args.patterns, args)
    late_scored, late_payload = score_late_labels(late_labels, args.late_model)
    suggestion_args = copy.copy(args)
    suggestion_args.max_turn = args.suggestion_max_turn
    suggestion_labels = build_labels_from_replays(args.patterns, suggestion_args)
    suggestion_scored, suggestion_payload = score_labels(suggestion_labels, args.suggestion_model)
    loss_events = replay_loss_events(args.patterns)

    late_top_night = top_late_per_night(late_scored)
    suggestion_top_night = top_per_night(suggestion_scored)
    late_raw_report, late_raw_summary = late_hit_report(late_scored, loss_events, args.lookahead_turns, args.big_loss_threshold)
    late_night_report, late_night_summary = late_hit_report(late_top_night, loss_events, args.lookahead_turns, args.big_loss_threshold)
    suggestion_night_report, suggestion_night_summary = dedup_summary(
        "suggestion_per_night_top1",
        suggestion_top_night,
        loss_events,
        args.lookahead_turns,
        args.big_loss_threshold,
    )
    overlap = overlap_report(late_top_night, suggestion_top_night, loss_events, args)
    combined_rate = float(overlap["combined_hit"].mean()) if len(overlap) else 0.0
    actionable_summary = thresholded_overlap_summary(
        overlap,
        args.late_action_threshold,
        args.suggestion_action_threshold,
    )

    write_csv(args.output_dir / "late_big_loss_scores.csv", late_scored)
    write_csv(args.output_dir / "late_per_night_top1_scores.csv", late_top_night)
    write_csv(args.output_dir / "suggestion_scores.csv", suggestion_scored)
    write_csv(args.output_dir / "suggestion_per_night_top1_scores.csv", suggestion_top_night)
    write_csv(args.output_dir / "late_big_loss_hit_report.csv", late_raw_report)
    write_csv(args.output_dir / "late_per_night_top1_hit_report.csv", late_night_report)
    write_csv(args.output_dir / "suggestion_per_night_top1_hit_report.csv", suggestion_night_report)
    write_csv(args.output_dir / "combined_big_loss_hit_report.csv", overlap)

    summary = {
        "patterns": args.patterns,
        "late_model": str(args.late_model),
        "suggestion_model": str(args.suggestion_model),
        "late_rows": int(len(late_scored)),
        "suggestion_rows": int(len(suggestion_scored)),
        "big_loss_threshold": args.big_loss_threshold,
        "lookahead_turns": args.lookahead_turns,
        "late_model_threshold": float(late_payload.get("threshold", 0.35)),
        "suggestion_model_threshold": float(suggestion_payload.get("threshold", 0.50)),
        "late_raw": late_raw_summary,
        "late_per_night_top1": late_night_summary,
        "suggestion_per_night_top1": suggestion_night_summary,
        "combined_per_night_top1": {
            "big_loss_events": int(len(overlap)),
            "combined_hit_rate": combined_rate,
            "late_only_hits": int(((overlap["late_hit"] == 1) & (overlap["suggestion_hit"] == 0)).sum()) if len(overlap) else 0,
            "suggestion_only_hits": int(((overlap["late_hit"] == 0) & (overlap["suggestion_hit"] == 1)).sum()) if len(overlap) else 0,
            "both_hits": int(overlap["both_hit"].sum()) if len(overlap) else 0,
            "misses": int((overlap["combined_hit"] == 0).sum()) if len(overlap) else 0,
        },
        "actionable_thresholded_per_night_top1": actionable_summary,
    }
    (args.output_dir / "combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
