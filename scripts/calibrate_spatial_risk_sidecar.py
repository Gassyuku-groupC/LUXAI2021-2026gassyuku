#!/usr/bin/env python3
"""Grouped replay calibration for the spatial risk sidecar, per map size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from imitation_learning_utils import load_agent_flags  # noqa: E402
from train_auxiliary_risk_head import batch_from_shard  # noqa: E402
from train_spatial_risk_sidecar import spatial_targets, stable_fraction  # noqa: E402
from lux_ai.nns import create_model  # noqa: E402
from lux_ai.rl_agent.auxiliary_heads import auxiliary_feature_forward  # noqa: E402
from lux_ai.rl_agent.spatial_risk_sidecar import SpatialRiskAttentionSidecar  # noqa: E402


def precision_threshold(labels: np.ndarray, scores: np.ndarray, target: float, minimum: int):
    order = np.argsort(-scores, kind="stable")
    labels = labels[order].astype(int)
    scores = scores[order]
    alerts = np.arange(1, len(labels) + 1)
    precision = np.cumsum(labels) / alerts
    ties = np.r_[scores[:-1] != scores[1:], True]
    eligible = ties & (alerts >= minimum) & (precision >= target)
    if not eligible.any():
        return 1.0, False
    indices = np.flatnonzero(eligible)
    recalls = np.cumsum(labels) / max(labels.sum(), 1)
    index = indices[np.argmax(recalls[indices])]
    return float(scores[index]), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, default=Path("outputs/submission_packages/best_agent"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spatial_risk_sidecar_v1/calibration"))
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--minimum-alerts", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    flags = load_agent_flags(args.agent_dir)
    actor = create_model(flags, device)
    actor.load_state_dict(torch.load(
        args.agent_dir / "lux_ai" / "rl_agent" / "candidate_weights.pt", map_location=device
    )["model_state_dict"], strict=True)
    actor.eval()
    sidecar = SpatialRiskAttentionSidecar(actor.base_out_channels).to(device)
    sidecar.load_state_dict(torch.load(args.checkpoint, map_location=device)["spatial_risk_sidecar_state_dict"])
    sidecar.eval()
    values = {size: {"score": [], "label": [], "groups": set()} for size in (12, 16, 24, 32)}
    batches = 0
    for shard_path in sorted(args.shards_dir.glob("shard_*.pt")):
        shard = torch.load(shard_path, map_location="cpu")
        by_map = {size: [] for size in values}
        for index, meta in enumerate(shard["meta"]):
            size = int(meta.get("width", meta.get("map_size", 0)) or 0)
            group = str(meta.get("episode_id") or meta.get("file"))
            if size in by_map and stable_fraction(group) >= 1.0 - args.validation_fraction:
                by_map[size].append(index)
                values[size]["groups"].add(group)
        for size, indices in by_map.items():
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                model_input, _ = batch_from_shard(shard, batch_indices, device)
                with torch.no_grad():
                    features, mask = auxiliary_feature_forward(actor, model_input)
                    output = sidecar(features.detach(), mask)
                labels, label_mask, _, _ = spatial_targets(shard, batch_indices, device)
                values[size]["score"].extend(output["risk_probabilities"][label_mask].cpu().tolist())
                values[size]["label"].extend(labels[label_mask].cpu().tolist())
                batches += 1
                if args.max_batches and batches >= args.max_batches:
                    break
            if args.max_batches and batches >= args.max_batches:
                break
        if args.max_batches and batches >= args.max_batches:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    thresholds = {}
    for size, data in values.items():
        labels = np.asarray(data["label"], dtype=int)
        scores = np.asarray(data["score"], dtype=float)
        if not len(labels) or not labels.sum():
            reports[str(size)] = {"samples": len(labels), "groups": len(data["groups"]), "calibrated": False}
            continue
        threshold, achieved = precision_threshold(labels, scores, args.target_precision, args.minimum_alerts)
        predictions = scores >= threshold
        tp = int((predictions & (labels == 1)).sum())
        fp = int((predictions & (labels == 0)).sum())
        fn = int((~predictions & (labels == 1)).sum())
        precision, recall, curve_thresholds = precision_recall_curve(labels, scores)
        map_dir = args.output_dir / f"map_{size}"
        map_dir.mkdir(exist_ok=True)
        pd.DataFrame({"threshold": np.r_[curve_thresholds, np.nan], "precision": precision, "recall": recall}).to_csv(map_dir / "precision_recall_curve.csv", index=False)
        reports[str(size)] = {
            "samples": int(len(labels)), "positive_samples": int(labels.sum()), "groups": len(data["groups"]),
            "threshold": threshold, "target_precision_achieved": achieved,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "average_precision": float(average_precision_score(labels, scores)),
        }
        thresholds[size] = threshold
    payload = {"target_precision": args.target_precision, "grouped_by_replay": True, "maps": reports}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "risk_thresholds.yaml").write_text(yaml.safe_dump({"maps": thresholds}, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
