#!/usr/bin/env python3
"""Train a non-intrusive city-collapse risk scorer from strategy features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


FEATURES = [
    "map_size",
    "turn",
    "turns_remaining",
    "cycle_turn",
    "pre_night",
    "is_night",
    "turns_to_night",
    "team",
    "rank",
    "city_tiles",
    "cities",
    "largest_city_size",
    "mean_city_size",
    "resource_near_cities",
    "isolated_cities_r3",
    "workers",
    "worker_citytile_ratio",
    "unit_cap_margin",
    "research",
    "fuel",
    "upkeep",
    "fuel_turns_total",
    "min_city_fuel_turns",
    "p25_city_fuel_turns",
    "median_city_fuel_turns",
    "mean_city_fuel_turns",
    "low_fuel_city_lt3",
    "low_fuel_city_lt5",
    "low_fuel_city_lt10",
    "unit_cargo_fuel",
    "action_count",
    "move_actions",
    "transfer_actions",
    "research_actions",
    "bw_actions",
    "bw_low_fuel_lt3_actions",
    "bw_low_fuel_lt5_actions",
    "bw_low_fuel_lt10_actions",
    "bcity_actions",
    "bcity_isolated_actions",
    "bcity_resource_near_actions",
    "bcity_adjacent_low_fuel_lt5_actions",
    "city_tiles_delta_next",
    "units_delta_next",
    "research_delta_next",
]


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def stable_split(row: dict, val_fraction: float) -> str:
    key = f"{row.get('file','')}:{row.get('episode_id','')}:{row.get('team','')}:{row.get('turn','')}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_fraction else "train"


def load_rows(paths: list[Path], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    splits: list[float] = []
    meta: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as in_file:
            for row in csv.DictReader(in_file):
                if args.map_size and int(as_float(row, "map_size")) != args.map_size:
                    continue
                label_value = as_float(row, args.label_column)
                y = 1.0 if label_value >= args.loss_threshold else 0.0
                xs.append([as_float(row, feature) for feature in FEATURES])
                ys.append(y)
                splits.append(1.0 if stable_split(row, args.val_fraction) == "val" else 0.0)
                meta.append({
                    "file": row.get("file", ""),
                    "turn": int(as_float(row, "turn")),
                    "team": int(as_float(row, "team")),
                    "label_value": label_value,
                })
                if args.max_rows and len(xs) >= args.max_rows:
                    break
        if args.max_rows and len(xs) >= args.max_rows:
            break
    if not xs:
        raise ValueError("No training rows loaded.")
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
        torch.tensor(splits, dtype=torch.bool),
        meta,
    )


class RiskMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize(x: torch.Tensor, train_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = x[~train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp(min=1e-6)
    return (x - mean) / std, mean, std


def auc_score(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    pairs = sorted(zip(y_score.tolist(), y_true.tolist()), key=lambda item: item[0])
    pos = sum(1 for _, y in pairs if y >= 0.5)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.0
    rank_sum = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y >= 0.5:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def metrics(y_true: torch.Tensor, prob: torch.Tensor, threshold: float) -> dict:
    pred = prob >= threshold
    y = y_true >= 0.5
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "n": int(y.numel()),
        "positive_rate": float(y.float().mean()),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score(y_true, prob),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    x, y, val_mask, _ = load_rows(args.features, args)
    x, mean, std = standardize(x, val_mask)
    train_mask = ~val_mask
    if val_mask.sum() == 0 or train_mask.sum() == 0:
        raise ValueError("Train/validation split is empty; adjust --val-fraction.")

    model = RiskMLP(x.shape[1], args.hidden, args.dropout)
    pos = y[train_mask].sum().clamp(min=1.0)
    neg = train_mask.sum() - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(neg / pos).clamp(min=1.0, max=args.max_pos_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = torch.where(train_mask)[0]
    best = {"val_auc": -1.0, "state": None, "epoch": 0}
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel())]
        total_loss = 0.0
        total_n = 0
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start:start + args.batch_size]
            logits = model(x[idx])
            loss = loss_fn(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * idx.numel()
            total_n += idx.numel()
        model.eval()
        with torch.no_grad():
            train_prob = torch.sigmoid(model(x[train_mask]))
            val_prob = torch.sigmoid(model(x[val_mask]))
        train_metrics = metrics(y[train_mask], train_prob, args.threshold)
        val_metrics = metrics(y[val_mask], val_prob, args.threshold)
        item = {
            "epoch": epoch,
            "loss": total_loss / max(total_n, 1),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(item)
        if val_metrics["auc"] > best["val_auc"]:
            best = {"val_auc": val_metrics["auc"], "state": model.state_dict(), "epoch": epoch}
        print(
            f"epoch={epoch} loss={item['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"val_precision={val_metrics['precision']:.4f}"
        )

    model.load_state_dict(best["state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": FEATURES,
            "mean": mean,
            "std": std,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "threshold": args.threshold,
        },
        args.output_dir / "risk_scorer.pt",
    )
    summary = {
        "features": [str(path) for path in args.features],
        "rows": int(x.shape[0]),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "label_column": args.label_column,
        "loss_threshold": args.loss_threshold,
        "decision_threshold": args.threshold,
        "best_epoch": best["epoch"],
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"model: {args.output_dir / 'risk_scorer.pt'}")
    print(f"summary: {args.output_dir / 'train_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a city-loss risk scorer.")
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostic_layer/risk_scorer_v1"))
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--label-column", default="future_team_loss_10")
    parser.add_argument("--loss-threshold", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
