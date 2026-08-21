#!/usr/bin/env python3
"""Build counterfactual training labels from combined diagnostic scores.

The output is intentionally label-only. It does not edit actor weights or
imitation indexes directly. Downstream training can decide how to consume the
penalty/reward values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "label_version",
    "label_source",
    "source_file",
    "episode_id",
    "map_size",
    "source_opponent",
    "team",
    "eval_side",
    "turn",
    "night_cycle",
    "action_scope",
    "unit_id",
    "action_taken",
    "suggested_action",
    "suggested_city_id",
    "late_prob",
    "suggestion_risk_score",
    "future_team_loss_20",
    "future_big_loss",
    "ignored_suggestion",
    "accepted_suggestion",
    "counterfactual_label",
    "penalty_label",
    "positive_label",
    "penalty_weight",
    "positive_weight",
    "reward_value",
    "training_note",
]


def num(series: pd.Series | object, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def as_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: pd.Series, key: str, default: int = 0) -> int:
    return int(as_float(row, key, float(default)))


def choose_score_file(report_dir: Path, name: str, use_all_scores: bool) -> Path:
    if use_all_scores:
        return report_dir / name
    if name == "late_big_loss_scores.csv":
        return report_dir / "late_per_night_top1_scores.csv"
    if name == "suggestion_scores.csv":
        return report_dir / "suggestion_per_night_top1_scores.csv"
    return report_dir / name.replace("_scores.csv", "_per_night_top1_scores.csv")


def source_file(row: pd.Series) -> str:
    return str(row.get("source_file") or row.get("file") or "")


def common_fields(row: pd.Series, args: argparse.Namespace, label_source: str) -> dict:
    return {
        "label_version": args.label_version,
        "label_source": label_source,
        "source_file": source_file(row),
        "episode_id": row.get("episode_id", ""),
        "map_size": row.get("map_size", ""),
        "source_opponent": row.get("source_opponent", ""),
        "team": row.get("team", ""),
        "eval_side": row.get("eval_side", ""),
        "turn": as_int(row, "turn"),
        "night_cycle": as_int(row, "night_cycle", as_int(row, "turn") // 40),
    }


def penalty_weight(loss: float, score: float, args: argparse.Namespace) -> float:
    weight = args.base_penalty_weight
    weight += min(loss / max(args.big_loss_threshold, 1), 4.0) * args.loss_penalty_scale
    weight += min(max(score, 0.0), 1.0) * args.score_penalty_scale
    return round(min(weight, args.max_penalty_weight), 4)


def suggestion_penalty_weight(loss: float, risk_score: float, args: argparse.Namespace) -> float:
    weight = args.base_penalty_weight
    weight += min(loss / max(args.big_loss_threshold, 1), 4.0) * args.loss_penalty_scale
    weight += min(max(risk_score, 0.0) / max(args.suggestion_action_threshold, 1e-6), 2.0) * args.suggestion_score_penalty_scale
    return round(min(weight, args.max_penalty_weight), 4)


def build_late_rows(late: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    rows = []
    if late.empty:
        return rows
    late = late.copy()
    late["pred_late_big_loss_prob"] = num(late.get("pred_late_big_loss_prob", 0.0))
    late["future_team_loss_20"] = num(late.get("future_team_loss_20", 0.0))
    actionable = late[late["pred_late_big_loss_prob"] >= args.late_action_threshold]
    for _, row in actionable.iterrows():
        loss = as_float(row, "future_team_loss_20")
        score = as_float(row, "pred_late_big_loss_prob")
        has_loss = int(loss >= args.big_loss_threshold)
        penalty = has_loss
        p_weight = penalty_weight(loss, score, args) if penalty else 0.0
        out = {
            **common_fields(row, args, "late_big_loss_warning"),
            "action_scope": "team_turn",
            "unit_id": "",
            "action_taken": "",
            "suggested_action": "reduce_macro_risk",
            "suggested_city_id": "",
            "late_prob": round(score, 6),
            "suggestion_risk_score": 0.0,
            "future_team_loss_20": round(loss, 4),
            "future_big_loss": has_loss,
            "ignored_suggestion": 1,
            "accepted_suggestion": 0,
            "counterfactual_label": "late_risk_then_big_loss" if has_loss else "late_risk_without_big_loss",
            "penalty_label": penalty,
            "positive_label": 0,
            "penalty_weight": p_weight,
            "positive_weight": 0.0,
            "reward_value": round(-p_weight, 4),
            "training_note": "macro warning; no direct action replacement",
        }
        rows.append(out)
    return rows


def build_suggestion_rows(suggestions: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    rows = []
    if suggestions.empty:
        return rows
    suggestions = suggestions.copy()
    suggestions["pred_risk_score"] = num(suggestions.get("pred_risk_score", 0.0))
    suggestions["future_team_loss_20"] = num(suggestions.get("future_team_loss_20", 0.0))
    actionable = suggestions[suggestions["pred_risk_score"] >= args.suggestion_action_threshold]
    for _, row in actionable.iterrows():
        loss = as_float(row, "future_team_loss_20")
        risk_score = as_float(row, "pred_risk_score")
        ignored = as_int(row, "ignored_suggestion", 1)
        accepted = 1 - ignored
        has_loss = int(loss >= args.big_loss_threshold)
        penalty = int(ignored and has_loss)
        positive = int(accepted and not has_loss)
        p_weight = suggestion_penalty_weight(loss, risk_score, args) if penalty else 0.0
        pos_weight = args.accepted_no_loss_positive_weight if positive else 0.0
        if penalty:
            label = "ignored_support_then_big_loss"
        elif ignored:
            label = "ignored_support_without_big_loss"
        elif positive:
            label = "accepted_support_without_big_loss"
        else:
            label = "accepted_support_but_big_loss"
        out = {
            **common_fields(row, args, "suggest_fuel_support"),
            "action_scope": "unit_action",
            "unit_id": row.get("unit_id", ""),
            "action_taken": row.get("action_taken", ""),
            "suggested_action": row.get("suggested_action", ""),
            "suggested_city_id": row.get("suggested_city_id", ""),
            "late_prob": 0.0,
            "suggestion_risk_score": round(risk_score, 6),
            "future_team_loss_20": round(loss, 4),
            "future_big_loss": has_loss,
            "ignored_suggestion": ignored,
            "accepted_suggestion": accepted,
            "counterfactual_label": label,
            "penalty_label": penalty,
            "positive_label": positive,
            "penalty_weight": p_weight,
            "positive_weight": pos_weight,
            "reward_value": round(pos_weight - p_weight, 4),
            "training_note": "local fuel support suggestion",
        }
        rows.append(out)
    return rows


def write_outputs(labels: pd.DataFrame, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.reindex(columns=OUTPUT_COLUMNS)
    labels.to_csv(args.output_dir / "counterfactual_risk_labels.csv", index=False, encoding="utf-8")
    summary = {
        "label_version": args.label_version,
        "report_dir": str(args.report_dir),
        "rows": int(len(labels)),
        "late_action_threshold": args.late_action_threshold,
        "suggestion_action_threshold": args.suggestion_action_threshold,
        "big_loss_threshold": args.big_loss_threshold,
    }
    if len(labels):
        summary.update(
            {
                "penalty_rows": int(pd.to_numeric(labels["penalty_label"], errors="coerce").fillna(0).sum()),
                "positive_rows": int(pd.to_numeric(labels["positive_label"], errors="coerce").fillna(0).sum()),
                "mean_reward_value": float(pd.to_numeric(labels["reward_value"], errors="coerce").fillna(0).mean()),
                "mean_penalty_weight": float(pd.to_numeric(labels["penalty_weight"], errors="coerce").fillna(0).mean()),
                "by_label_source": labels.groupby("label_source").size().to_dict(),
                "by_counterfactual_label": labels.groupby("counterfactual_label").size().to_dict(),
                "penalty_by_source": labels.groupby("label_source")["penalty_label"].sum().to_dict(),
            }
        )
    (args.output_dir / "counterfactual_risk_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build counterfactual risk labels from a combined risk report directory.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/counterfactual_risk_labels_v1"))
    parser.add_argument("--label-version", default="counterfactual_risk_v1")
    parser.add_argument("--use-all-scores", action="store_true", help="Use all score rows instead of per-night top1 rows.")
    parser.add_argument("--late-action-threshold", type=float, default=0.35)
    parser.add_argument("--suggestion-action-threshold", type=float, default=2.0)
    parser.add_argument("--big-loss-threshold", type=float, default=10.0)
    parser.add_argument("--base-penalty-weight", type=float, default=1.0)
    parser.add_argument("--loss-penalty-scale", type=float, default=0.35)
    parser.add_argument("--score-penalty-scale", type=float, default=0.50)
    parser.add_argument("--suggestion-score-penalty-scale", type=float, default=0.35)
    parser.add_argument("--max-penalty-weight", type=float, default=3.0)
    parser.add_argument("--accepted-no-loss-positive-weight", type=float, default=0.10)
    args = parser.parse_args()

    late_path = choose_score_file(args.report_dir, "late_big_loss_scores.csv", args.use_all_scores)
    suggestion_path = choose_score_file(args.report_dir, "suggestion_scores.csv", args.use_all_scores)
    if not late_path.exists():
        raise FileNotFoundError(late_path)
    if not suggestion_path.exists():
        raise FileNotFoundError(suggestion_path)
    late = pd.read_csv(late_path)
    suggestions = pd.read_csv(suggestion_path)
    labels = pd.DataFrame([*build_late_rows(late, args), *build_suggestion_rows(suggestions, args)])
    write_outputs(labels, args)


if __name__ == "__main__":
    main()
