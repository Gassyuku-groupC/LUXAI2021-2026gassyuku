#!/usr/bin/env python3
"""Score candidate actions for best-agent replay states with v2 scorers.

For each state row, this script creates four counterfactual candidate rows:
no_expand, bw, bcity, and research. It scores each row with the v2 diagnostic
models and emits offline suggestions. It does not modify agent behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_strategy_candidate_scorers_v2 import FEATURES, SCORER_SPECS, add_candidate_features  # noqa: E402
from train_strategy_label_scorers import numeric_eval_side  # noqa: E402


ACTIONS = ["no_expand", "bw", "bcity", "research"]
SCORE_COLUMNS = {
    "candidate_risk_city_loss_20": "p_risk_city_loss_20",
    "candidate_risk_big_loss_20": "p_risk_big_loss_20",
    "candidate_error_failed_big_loss": "p_error_failed_big_loss",
    "candidate_safe_expansion_success_40": "p_safe_expansion_success_40",
    "candidate_success_stable_scale": "p_success_stable_scale",
}


def parse_csv_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def load_base_rows(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.input_csv, low_memory=False)
    if args.map_size:
        data = data[pd.to_numeric(data["map_size"], errors="coerce").fillna(0).astype(int) == args.map_size].copy()
    if args.source_kind:
        data = data[data["source_kind"].fillna("").astype(str).isin(parse_csv_set(args.source_kind))].copy()
    if args.team_names:
        data = data[data["team_name"].fillna("").astype(str).isin(parse_csv_set(args.team_names))].copy()
    for needle in args.file_contains:
        data = data[data["file"].fillna("").astype(str).str.contains(needle, regex=False)].copy()
    if args.candidate_only:
        side = numeric_eval_side(data.get("eval_side", pd.Series([""] * len(data))))
        team = pd.to_numeric(data["team"], errors="coerce").fillna(-999).astype(int)
        data = data[(side.astype(int) >= 0) & (team == side.astype(int))].copy()
    if args.max_rows and len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    if data.empty:
        raise ValueError("No rows left after filters.")
    data = add_candidate_features(data)
    data.rename(columns={"candidate_action": "actual_action"}, inplace=True)
    data["state_id"] = range(len(data))
    return data


def make_candidate_rows(base: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for action in ACTIONS:
        frame = base.copy()
        frame["candidate_action"] = action
        for candidate in ACTIONS:
            frame[f"candidate_is_{candidate}"] = int(action == candidate)
        frame["candidate_is_low_fuel_bw"] = (
            (action == "bw")
            & (pd.to_numeric(frame.get("low_fuel_city_lt5", 0), errors="coerce").fillna(0) > 0)
        ).astype(int)
        frame["candidate_is_low_fuel_bcity"] = (
            (action == "bcity")
            & (pd.to_numeric(frame.get("low_fuel_city_lt5", 0), errors="coerce").fillna(0) > 0)
        ).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_features(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    if "eval_side_numeric" in features and "eval_side_numeric" not in frame:
        frame["eval_side_numeric"] = numeric_eval_side(frame.get("eval_side", pd.Series([""] * len(frame))))
    for feature in features:
        if feature not in frame:
            frame[feature] = 0.0
    return frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def score_candidates(candidates: pd.DataFrame, model_dir: Path) -> tuple[pd.DataFrame, dict]:
    scored = candidates.copy()
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


def choose_best_actions(scored: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for _, group in scored.groupby("state_id", sort=False):
        base = group.iloc[0].to_dict()
        actual = str(base.get("actual_action", ""))
        group = group.copy()
        group["bad_score"] = (
            group["p_risk_big_loss_20"] * args.big_loss_weight
            + group["p_error_failed_big_loss"] * args.error_weight
            + group["p_risk_city_loss_20"] * args.loss_weight
            - group["p_success_stable_scale"] * args.success_credit
        )
        actual_row = group[group["candidate_action"] == actual]
        if actual_row.empty:
            actual_row = group[group["candidate_action"] == "no_expand"]
        actual_row = actual_row.iloc[0]
        best_risk = group.sort_values(["bad_score", "p_risk_big_loss_20"], ascending=[True, True]).iloc[0]
        best_expand = group.sort_values(
            ["p_safe_expansion_success_40", "p_risk_big_loss_20", "p_error_failed_big_loss"],
            ascending=[False, True, True],
        ).iloc[0]

        suggest = "keep_actual"
        reason = "actual action is near the best scored candidate"
        risk_delta = float(actual_row["bad_score"] - best_risk["bad_score"])
        if (
            risk_delta >= args.min_bad_score_delta
            and float(actual_row["p_risk_big_loss_20"]) >= args.high_big_risk_threshold
            and (
                float(actual_row["p_risk_big_loss_20"] - best_risk["p_risk_big_loss_20"]) >= args.min_risk_prob_delta
                or float(actual_row["p_error_failed_big_loss"] - best_risk["p_error_failed_big_loss"]) >= args.min_error_prob_delta
            )
            and str(best_risk["candidate_action"]) != "bcity"
            and str(best_risk["candidate_action"]) != actual
        ):
            suggest = f"suggest_{best_risk['candidate_action']}"
            reason = "actual action has materially higher predicted risk"
        elif (
            actual == "no_expand"
            and str(best_expand["candidate_action"]) == "bcity"
            and float(best_expand["p_safe_expansion_success_40"]) >= args.safe_expansion_threshold
            and float(best_expand["p_risk_big_loss_20"]) <= args.low_big_risk_threshold
        ):
            suggest = "missed_safe_bcity_window"
            reason = "bcity candidate has high safe-expansion score and low big-loss risk"

        out = {
            "state_id": int(base["state_id"]),
            "file": base.get("file", ""),
            "source_opponent": base.get("source_opponent", ""),
            "eval_side": base.get("eval_side", ""),
            "team": base.get("team", ""),
            "team_name": base.get("team_name", ""),
            "opponent_name": base.get("opponent_name", ""),
            "rank": base.get("rank", ""),
            "turn": base.get("turn", ""),
            "phase": base.get("phase", ""),
            "cycle_turn": base.get("cycle_turn", ""),
            "turns_to_night": base.get("turns_to_night", ""),
            "city_tiles": base.get("city_tiles", ""),
            "workers": base.get("workers", ""),
            "research": base.get("research", ""),
            "min_city_fuel_turns": base.get("min_city_fuel_turns", ""),
            "p25_city_fuel_turns": base.get("p25_city_fuel_turns", ""),
            "low_fuel_city_lt5": base.get("low_fuel_city_lt5", ""),
            "future_team_loss_20": base.get("future_team_loss_20", ""),
            "final_city_tile_margin": base.get("final_city_tile_margin", ""),
            "actual_action": actual,
            "best_risk_action": best_risk["candidate_action"],
            "best_expand_action": best_expand["candidate_action"],
            "suggestion": suggest,
            "reason": reason,
            "actual_bad_score": float(actual_row["bad_score"]),
            "best_bad_score": float(best_risk["bad_score"]),
            "bad_score_delta": risk_delta,
            "actual_big_risk": float(actual_row["p_risk_big_loss_20"]),
            "best_big_risk": float(best_risk["p_risk_big_loss_20"]),
            "actual_error_risk": float(actual_row["p_error_failed_big_loss"]),
            "best_error_risk": float(best_risk["p_error_failed_big_loss"]),
            "actual_safe_expansion": float(actual_row["p_safe_expansion_success_40"]),
            "best_expand_safe_expansion": float(best_expand["p_safe_expansion_success_40"]),
            "best_expand_big_risk": float(best_expand["p_risk_big_loss_20"]),
        }
        for _, candidate in group.iterrows():
            action = candidate["candidate_action"]
            out[f"{action}_big_risk"] = float(candidate["p_risk_big_loss_20"])
            out[f"{action}_error_risk"] = float(candidate["p_error_failed_big_loss"])
            out[f"{action}_safe_expansion"] = float(candidate["p_safe_expansion_success_40"])
            out[f"{action}_success"] = float(candidate["p_success_stable_scale"])
        rows.append(out)
    return pd.DataFrame(rows)


def write_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8")


def write_summary(path: Path, suggestions: pd.DataFrame, scored: pd.DataFrame, metadata: dict, args: argparse.Namespace) -> None:
    actionable = suggestions[suggestions["suggestion"] != "keep_actual"].copy()
    summary = {
        "input_csv": str(args.input_csv),
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "states": int(len(suggestions)),
        "candidate_rows": int(len(scored)),
        "actionable_suggestions": int(len(actionable)),
        "suggestion_counts": suggestions["suggestion"].value_counts().to_dict(),
        "actual_action_counts": suggestions["actual_action"].value_counts().to_dict(),
        "by_source_opponent": suggestions.groupby("source_opponent")["suggestion"].value_counts().unstack(fill_value=0).to_dict("index"),
        "by_rank": suggestions.groupby("rank")["suggestion"].value_counts().unstack(fill_value=0).to_dict("index"),
        "scorers": metadata,
        "thresholds": {
            "safe_expansion_threshold": args.safe_expansion_threshold,
            "high_big_risk_threshold": args.high_big_risk_threshold,
            "low_big_risk_threshold": args.low_big_risk_threshold,
            "min_bad_score_delta": args.min_bad_score_delta,
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score four candidate actions per best-agent state.")
    parser.add_argument("--input-csv", type=Path, default=Path("dataset/processed/strategy_label_dataset_v1.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/diagnostic_layer/strategy_candidate_scorers_v2_16"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/best_agent_candidate_action_suggestions_v2_16"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--source-kind", default="local_eval")
    parser.add_argument("--team-names", default="")
    parser.add_argument("--file-contains", action="append", default=[])
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--safe-expansion-threshold", type=float, default=0.80)
    parser.add_argument("--high-big-risk-threshold", type=float, default=0.25)
    parser.add_argument("--low-big-risk-threshold", type=float, default=0.12)
    parser.add_argument("--min-bad-score-delta", type=float, default=0.08)
    parser.add_argument("--min-risk-prob-delta", type=float, default=0.05)
    parser.add_argument("--min-error-prob-delta", type=float, default=0.03)
    parser.add_argument("--big-loss-weight", type=float, default=1.0)
    parser.add_argument("--error-weight", type=float, default=0.8)
    parser.add_argument("--loss-weight", type=float, default=0.25)
    parser.add_argument("--success-credit", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()

    base = load_base_rows(args)
    candidates = make_candidate_rows(base)
    scored, metadata = score_candidates(candidates, args.model_dir)
    suggestions = choose_best_actions(scored, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_action_scores.csv", scored)
    write_csv(args.output_dir / "candidate_action_suggestions.csv", suggestions)
    write_csv(
        args.output_dir / "actionable_suggestions.csv",
        suggestions[suggestions["suggestion"] != "keep_actual"].sort_values("bad_score_delta", ascending=False),
    )
    write_summary(args.output_dir / "candidate_action_suggestion_summary.json", suggestions, scored, metadata, args)
    print(json.dumps({
        "states": int(len(suggestions)),
        "candidate_rows": int(len(scored)),
        "suggestion_counts": suggestions["suggestion"].value_counts().to_dict(),
        "output": str(args.output_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
