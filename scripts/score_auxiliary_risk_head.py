#!/usr/bin/env python3
"""Score a trained auxiliary risk head on imitation shards."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_auxiliary_risk_head import batch_from_shard, labels_from_meta, select_player  # noqa: E402
from imitation_learning_utils import load_agent_flags  # noqa: E402
from lux_ai.nns import create_model  # noqa: E402
from lux_ai.rl_agent.auxiliary_heads import AuxiliaryRiskHead, auxiliary_feature_forward  # noqa: E402


def average_precision(scores: List[float], labels: List[float]) -> float | None:
    positives = sum(1 for label in labels if label > 0.5)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hit = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] > 0.5:
            hit += 1
            precision_sum += hit / rank
    return precision_sum / positives


def roc_auc(scores: List[float], labels: List[float]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label > 0.5]
    negatives = [score for score, label in zip(scores, labels) if label <= 0.5]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def threshold_stats(scores: List[float], labels: List[float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        pred = score >= threshold
        truth = label > 0.5
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "accuracy": accuracy}


def summarize_task(scores: List[float], labels: List[float], thresholds: List[float]) -> dict:
    if not labels:
        return {}
    bce = F.binary_cross_entropy(
        torch.tensor(scores, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6),
        torch.tensor(labels, dtype=torch.float32),
    ).item()
    return {
        "count": len(labels),
        "positive_rate": sum(labels) / len(labels),
        "mean_score": sum(scores) / len(scores),
        "bce": bce,
        "auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "thresholds": {
            str(threshold): threshold_stats(scores, labels, threshold)
            for threshold in thresholds
        },
    }


def cached_labels_for_indices(label_shard: dict, indices: list[int], device: torch.device) -> dict:
    idx = torch.tensor(indices, dtype=torch.long)
    return {
        "players": label_shard["players"].index_select(0, idx).to(device, non_blocking=True),
        "loss10": label_shard["loss10"].index_select(0, idx).to(device, non_blocking=True),
        "loss20": label_shard["loss20"].index_select(0, idx).to(device, non_blocking=True),
        "unsafe_large_city": label_shard["unsafe_large_city"].index_select(0, idx).to(device, non_blocking=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an auxiliary risk head.")
    parser.add_argument("--head-path", type=Path, default=Path("outputs/auxiliary_risk_head_v1/auxiliary_risk_head.pt"))
    parser.add_argument("--shards-dir", type=Path, default=Path("dataset/processed/imitation_shards_counterfactual_v4_residual"))
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/processed/auxiliary_labels_counterfactual_v4_residual"))
    parser.add_argument("--agent-dir", type=Path, default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"))
    parser.add_argument("--output", type=Path, default=Path("outputs/auxiliary_risk_head_v1/score_summary.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    random.seed(args.seed)
    thresholds = [
        float(part.strip())
        for part in args.thresholds.split(",")
        if part.strip()
    ]
    if args.threshold not in thresholds:
        thresholds.append(args.threshold)
    thresholds = sorted(set(thresholds))
    device = torch.device(args.device)
    flags = load_agent_flags(args.agent_dir)
    model = create_model(flags, device)
    checkpoint = torch.load(args.agent_dir / "lux_ai" / "rl_agent" / "candidate_weights.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    head_checkpoint = torch.load(args.head_path, map_location=device)
    aux_head = AuxiliaryRiskHead(int(head_checkpoint.get("in_channels", model.base_out_channels))).to(device)
    aux_head.load_state_dict(head_checkpoint["aux_head_state_dict"])
    aux_head.eval()

    task_scores: Dict[str, List[float]] = {"loss10": [], "loss20": [], "unsafe_large_city": []}
    task_labels: Dict[str, List[float]] = {"loss10": [], "loss20": [], "unsafe_large_city": []}
    shard_paths = sorted(args.shards_dir.glob("shard_*.pt"))
    seen = 0
    with torch.no_grad():
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu")
            label_shard = None
            label_path = args.labels_dir / shard_path.name
            if label_path.exists():
                label_shard = torch.load(label_path, map_location="cpu")
            n = len(shard["meta"])
            indices = list(range(n))
            for start in range(0, n, args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                if args.max_samples and seen >= args.max_samples:
                    break
                if args.max_samples:
                    batch_indices = batch_indices[: max(args.max_samples - seen, 0)]
                model_input, meta = batch_from_shard(shard, batch_indices, device)
                if label_shard is None:
                    labels = labels_from_meta(meta, device)
                else:
                    labels = cached_labels_for_indices(label_shard, batch_indices, device)
                actor_features, input_mask = auxiliary_feature_forward(model, model_input)
                outputs = aux_head(actor_features, input_mask)
                for task, key in (
                    ("loss10", "loss10_logit"),
                    ("loss20", "loss20_logit"),
                    ("unsafe_large_city", "unsafe_large_city_logit"),
                ):
                    logits = select_player(outputs[key], labels["players"])
                    probs = torch.sigmoid(logits).detach().cpu().tolist()
                    values = labels[task].detach().cpu().tolist()
                    task_scores[task].extend(float(x) for x in probs)
                    task_labels[task].extend(float(x) for x in values)
                seen += len(batch_indices)
            if args.max_samples and seen >= args.max_samples:
                break

    summary = {
        "head_path": str(args.head_path),
        "shards_dir": str(args.shards_dir),
        "labels_dir": str(args.labels_dir),
        "samples": seen,
        "threshold": args.threshold,
        "thresholds": thresholds,
        "tasks": {
            task: summarize_task(task_scores[task], task_labels[task], thresholds)
            for task in task_scores
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
