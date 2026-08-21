#!/usr/bin/env python3
"""Score Lux strategy-label rows with trained diagnostic scorers.

This is intended for offline replay review of the current best agent. It writes
per-turn probabilities plus compact reports for top risk/error turns and missed
safe-expansion windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_strategy_label_scorers import FEATURES, SCORER_SPECS, numeric_eval_side  # noqa: E402


SCORE_COLUMNS = {
    "risk_city_loss_20": "p_risk_city_loss_20",
    "risk_big_loss_20": "p_risk_big_loss_20",
    "error_failed_big_loss": "p_error_failed_big_loss",
    "safe_expansion_success_40": "p_safe_expansion_success_40",
    "success_stable_scale": "p_success_stable_scale",
}

REPORT_COLUMNS = [
    "file",
    "source_kind",
    "source_opponent",
    "eval_side",
    "team",
    "team_name",
    "opponent_name",
    "rank",
    "turn",
    "phase",
    "cycle_turn",
    "turns_to_night",
    "city_tiles",
    "workers",
    "research",
    "fuel_turns_total",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "low_fuel_city_lt5",
    "low_fuel_city_lt10",
    "bw_actions",
    "bw_low_fuel_lt5_actions",
    "bcity_actions",
    "bcity_adjacent_low_fuel_lt5_actions",
    "research_actions",
    "future_team_loss_10",
    "future_team_loss_20",
    "future_team_loss_40",
    "future_city_tiles_gain_40",
    "final_city_tiles",
    "final_opponent_city_tiles",
    "final_city_tile_margin",
    "strategy_label",
    *SCORE_COLUMNS.values(),
]


def parse_csv_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.input_csv, low_memory=False)
    if args.map_size:
        data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == args.map_size].copy()
    if args.source_kind:
        allowed = parse_csv_set(args.source_kind)
        data = data[data["source_kind"].fillna("").astype(str).isin(allowed)].copy()
    if args.team_names:
        allowed_teams = parse_csv_set(args.team_names)
        data = data[data["team_name"].fillna("").astype(str).isin(allowed_teams)].copy()
    for needle in args.file_contains:
        data = data[data["file"].fillna("").astype(str).str.contains(needle, regex=False)].copy()
    if args.candidate_only:
        data["eval_side_numeric"] = numeric_eval_side(data.get("eval_side", pd.Series([""] * len(data))))
        team = pd.to_numeric(data["team"], errors="coerce").fillna(-999).astype(int)
        side = pd.to_numeric(data["eval_side_numeric"], errors="coerce").fillna(-1).astype(int)
        data = data[(side >= 0) & (team == side)].copy()
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if data.empty:
        raise ValueError("No rows left after filters.")
    return data


def prepare_features(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    if "eval_side_numeric" in features and "eval_side_numeric" not in frame:
        frame["eval_side_numeric"] = numeric_eval_side(frame.get("eval_side", pd.Series([""] * len(frame))))
    for feature in features:
        if feature not in frame:
            frame[feature] = 0.0
    return frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def score(data: pd.DataFrame, model_dir: Path) -> tuple[pd.DataFrame, dict]:
    scored = data.copy()
    metadata = {}
    for scorer_name, score_column in SCORE_COLUMNS.items():
        model_path = model_dir / scorer_name / f"{scorer_name}_lgbm.joblib"
        payload = joblib.load(model_path)
        model = payload["model"]
        features = payload.get("features", FEATURES)
        scored[score_column] = model.predict_proba(prepare_features(scored, features))[:, 1]
        metadata[scorer_name] = {
            "model_path": str(model_path),
            "threshold": float(payload.get("threshold", SCORER_SPECS[scorer_name]["threshold"])),
            "label_column": payload.get("label_column", SCORER_SPECS[scorer_name]["label"]),
        }
    return scored, metadata


def write_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8")


def report_frame(data: pd.DataFrame) -> pd.DataFrame:
    cols = [column for column in REPORT_COLUMNS if column in data.columns]
    return data[cols].copy()


def top_by_episode(data: pd.DataFrame, score_column: str, top_n: int) -> pd.DataFrame:
    rows = []
    sort_cols = ["file", "team", score_column]
    ranked = data.sort_values(sort_cols, ascending=[True, True, False])
    for _, group in ranked.groupby(["file", "team"], dropna=False, sort=False):
        rows.append(group.head(top_n))
    if not rows:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    return report_frame(pd.concat(rows, ignore_index=True))


def missed_safe_expansion(data: pd.DataFrame, threshold: float, top_n: int) -> pd.DataFrame:
    frame = data.copy()
    for column in ["bcity_actions", "turns_remaining", "unit_cap_margin"]:
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce").fillna(0)
    mask = (
        (frame["p_safe_expansion_success_40"] >= threshold)
        & (frame["bcity_actions"] <= 0)
        & (frame["turns_remaining"] >= 40)
        & (frame["unit_cap_margin"] > 0)
    )
    return top_by_episode(frame[mask].copy(), "p_safe_expansion_success_40", top_n)


def risky_action_turns(data: pd.DataFrame, risk_threshold: float, error_threshold: float, top_n: int) -> pd.DataFrame:
    frame = data.copy()
    action_cols = ["bw_actions", "bcity_actions", "research_actions", "bw_low_fuel_lt5_actions", "bcity_adjacent_low_fuel_lt5_actions"]
    for column in action_cols:
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce").fillna(0)
    has_action = frame[action_cols].sum(axis=1) > 0
    mask = has_action & (
        (frame["p_risk_big_loss_20"] >= risk_threshold)
        | (frame["p_error_failed_big_loss"] >= error_threshold)
    )
    scored = frame[mask].copy()
    scored["p_combined_bad"] = scored[["p_risk_big_loss_20", "p_error_failed_big_loss"]].max(axis=1)
    return top_by_episode(scored, "p_combined_bad", top_n)


def episode_summary(data: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    numeric_cols = [
        *SCORE_COLUMNS.values(),
        "future_team_loss_20",
        "future_team_loss_40",
        "final_city_tiles",
        "final_opponent_city_tiles",
        "final_city_tile_margin",
    ]
    frame = data.copy()
    for column in numeric_cols:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    rows = []
    for keys, group in frame.groupby(["file", "team"], dropna=False):
        file_value, team = keys
        row = {
            "file": file_value,
            "team": int(team),
            "team_name": str(group["team_name"].iloc[0]) if "team_name" in group else "",
            "opponent_name": str(group["opponent_name"].iloc[0]) if "opponent_name" in group else "",
            "source_opponent": str(group["source_opponent"].iloc[0]) if "source_opponent" in group else "",
            "eval_side": str(group["eval_side"].iloc[0]) if "eval_side" in group else "",
            "rank": group["rank"].iloc[0] if "rank" in group else "",
            "turn_rows": int(len(group)),
            "max_risk_city_loss_20": float(group["p_risk_city_loss_20"].max()),
            "max_risk_big_loss_20": float(group["p_risk_big_loss_20"].max()),
            "max_error_failed_big_loss": float(group["p_error_failed_big_loss"].max()),
            "max_safe_expansion_success_40": float(group["p_safe_expansion_success_40"].max()),
            "mean_success_stable_scale": float(group["p_success_stable_scale"].mean()),
            "actual_max_future_loss20": float(group["future_team_loss_20"].max()) if "future_team_loss_20" in group else 0.0,
            "actual_max_future_loss40": float(group["future_team_loss_40"].max()) if "future_team_loss_40" in group else 0.0,
            "final_city_tiles": float(group["final_city_tiles"].iloc[-1]) if "final_city_tiles" in group else 0.0,
            "final_opponent_city_tiles": float(group["final_opponent_city_tiles"].iloc[-1]) if "final_opponent_city_tiles" in group else 0.0,
            "final_city_tile_margin": float(group["final_city_tile_margin"].iloc[-1]) if "final_city_tile_margin" in group else 0.0,
        }
        for scorer_name, score_column in SCORE_COLUMNS.items():
            threshold = metadata[scorer_name]["threshold"]
            row[f"{scorer_name}_alert_rate"] = float((group[score_column] >= threshold).mean())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["max_error_failed_big_loss", "max_risk_big_loss_20", "actual_max_future_loss20"],
        ascending=[False, False, False],
    )
    overall = {
        "episodes": int(len(summary)),
        "rows": int(len(data)),
        "mean_max_risk_city_loss_20": float(summary["max_risk_city_loss_20"].mean()) if len(summary) else 0.0,
        "mean_max_risk_big_loss_20": float(summary["max_risk_big_loss_20"].mean()) if len(summary) else 0.0,
        "mean_max_error_failed_big_loss": float(summary["max_error_failed_big_loss"].mean()) if len(summary) else 0.0,
        "mean_max_safe_expansion_success_40": float(summary["max_safe_expansion_success_40"].mean()) if len(summary) else 0.0,
        "mean_success_stable_scale": float(summary["mean_success_stable_scale"].mean()) if len(summary) else 0.0,
        "mean_actual_max_future_loss20": float(summary["actual_max_future_loss20"].mean()) if len(summary) else 0.0,
    }
    return summary, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Score strategy-label dataset rows with trained diagnostic scorers.")
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/diagnostic_layer/strategy_label_scorers_v1_16"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/best_agent_strategy_scores_v1_16"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--source-kind", default="local_eval", help="Comma-separated source kinds. Empty keeps all.")
    parser.add_argument("--team-names", default="", help="Comma-separated team names. Empty keeps all.")
    parser.add_argument("--file-contains", action="append", default=[], help="Keep rows whose file path contains this text. Repeatable.")
    parser.add_argument("--candidate-only", action="store_true", help="For local eval replays, keep only the evaluated agent side from _p0/_p1.")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--safe-expansion-threshold", type=float, default=0.60)
    parser.add_argument("--risky-action-threshold", type=float, default=0.35)
    parser.add_argument("--error-action-threshold", type=float, default=0.20)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    data = load_rows(args)
    scored, metadata = score(data, args.model_dir)
    episode_table, overall = episode_summary(scored, metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scored_turns.csv", scored)
    write_csv(args.output_dir / "episode_summary.csv", episode_table)
    write_csv(args.output_dir / "top_risk_city_loss_20_turns.csv", top_by_episode(scored, "p_risk_city_loss_20", args.top_n))
    write_csv(args.output_dir / "top_risk_big_loss_20_turns.csv", top_by_episode(scored, "p_risk_big_loss_20", args.top_n))
    write_csv(args.output_dir / "top_error_failed_big_loss_turns.csv", top_by_episode(scored, "p_error_failed_big_loss", args.top_n))
    write_csv(args.output_dir / "top_safe_expansion_windows.csv", top_by_episode(scored, "p_safe_expansion_success_40", args.top_n))
    write_csv(args.output_dir / "missed_safe_expansion_windows.csv", missed_safe_expansion(scored, args.safe_expansion_threshold, args.top_n))
    write_csv(args.output_dir / "risky_action_turns.csv", risky_action_turns(scored, args.risky_action_threshold, args.error_action_threshold, args.top_n))

    meta = {
        "input_csv": str(args.input_csv),
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "filters": {
            "map_size": args.map_size,
            "source_kind": args.source_kind,
            "team_names": args.team_names,
            "file_contains": args.file_contains,
            "candidate_only": bool(args.candidate_only),
        },
        "scorers": metadata,
        "overall": overall,
        "files": {
            "scored_turns": str(args.output_dir / "scored_turns.csv"),
            "episode_summary": str(args.output_dir / "episode_summary.csv"),
            "top_risk_city_loss_20_turns": str(args.output_dir / "top_risk_city_loss_20_turns.csv"),
            "top_risk_big_loss_20_turns": str(args.output_dir / "top_risk_big_loss_20_turns.csv"),
            "top_error_failed_big_loss_turns": str(args.output_dir / "top_error_failed_big_loss_turns.csv"),
            "top_safe_expansion_windows": str(args.output_dir / "top_safe_expansion_windows.csv"),
            "missed_safe_expansion_windows": str(args.output_dir / "missed_safe_expansion_windows.csv"),
            "risky_action_turns": str(args.output_dir / "risky_action_turns.csv"),
        },
    }
    (args.output_dir / "score_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
