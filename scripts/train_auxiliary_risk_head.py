#!/usr/bin/env python3
"""Train a frozen-backbone auxiliary risk head from imitation shards."""

from __future__ import annotations

import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imitation_learning_utils import load_agent_flags  # noqa: E402
from lux_ai.nns import create_model  # noqa: E402
from lux_ai.rl_agent.auxiliary_heads import AuxiliaryRiskHead, auxiliary_feature_forward  # noqa: E402


def to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True)


def iter_batches(n: int, batch_size: int, shuffle: bool) -> Iterable[List[int]]:
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start: start + batch_size]


def batch_from_shard(shard: dict, indices: Iterable[int], device: torch.device) -> tuple[dict, list[dict]]:
    idx = torch.tensor(list(indices), dtype=torch.long)
    obs = {key: value.index_select(0, idx) for key, value in shard["obs"].items()}
    available = {
        key: value.index_select(0, idx) for key, value in shard["available_actions_mask"].items()
    }
    model_input = {
        "obs": to_device(obs, device),
        "info": {
            "input_mask": shard["input_mask"].index_select(0, idx).to(device, non_blocking=True),
            "available_actions_mask": to_device(available, device),
        },
    }
    meta = [shard["meta"][int(i)] for i in idx.tolist()]
    return model_input, meta


@lru_cache(maxsize=64)
def load_replay_timeline(path_text: str) -> dict:
    path = Path(path_text)
    with path.open(encoding="utf-8") as replay_file:
        replay = json.load(replay_file)
    steps = replay.get("steps") or []
    city_counts = []
    unsafe_large = []
    for step in steps:
        updates = []
        if step and step[0].get("observation"):
            updates = step[0]["observation"].get("updates") or []
        counts = [0, 0]
        city_fuel = [{}, {}]
        city_upkeep = [{}, {}]
        city_tiles = [{}, {}]
        for update in updates:
            parts = update.split()
            if not parts:
                continue
            if parts[0] == "ct" and len(parts) >= 4:
                team = int(parts[1])
                city_id = parts[2]
                counts[team] += 1
                city_tiles[team][city_id] = city_tiles[team].get(city_id, 0) + 1
            elif parts[0] == "c" and len(parts) >= 5:
                team = int(parts[1])
                city_id = parts[2]
                city_fuel[team][city_id] = float(parts[3])
                city_upkeep[team][city_id] = float(parts[4])
        city_counts.append(counts)
        unsafe_large.append([
            any(
                tiles >= 20
                and city_fuel[team].get(city_id, 0.0) / max(city_upkeep[team].get(city_id, 1.0), 1.0) < 12.0
                for city_id, tiles in city_tiles[team].items()
            )
            for team in (0, 1)
        ])
    return {"city_counts": city_counts, "unsafe_large": unsafe_large}


def labels_from_meta(meta: list[dict], device: torch.device) -> dict:
    loss10 = []
    loss20 = []
    unsafe_large = []
    players = []
    for item in meta:
        timeline = load_replay_timeline(str(item["file"]))
        step = int(item["state_step"])
        player = int(item["teacher_player"])
        players.append(player)
        counts = timeline["city_counts"]
        current = counts[min(step, len(counts) - 1)][player] if counts else 0
        future10 = counts[step + 1: min(step + 11, len(counts))]
        future20 = counts[step + 1: min(step + 21, len(counts))]
        min10 = min([row[player] for row in future10], default=current)
        min20 = min([row[player] for row in future20], default=current)
        loss10.append(float(min10 < current))
        loss20.append(float(min20 < current))
        unsafe = timeline["unsafe_large"][min(step, len(timeline["unsafe_large"]) - 1)][player] if timeline["unsafe_large"] else False
        unsafe_large.append(float(unsafe))
    return {
        "players": torch.tensor(players, dtype=torch.long, device=device),
        "loss10": torch.tensor(loss10, dtype=torch.float32, device=device),
        "loss20": torch.tensor(loss20, dtype=torch.float32, device=device),
        "unsafe_large_city": torch.tensor(unsafe_large, dtype=torch.float32, device=device),
    }


def cached_labels_for_indices(label_shard: dict, indices: list[int], device: torch.device) -> dict:
    idx = torch.tensor(indices, dtype=torch.long)
    return {
        "players": label_shard["players"].index_select(0, idx).to(device, non_blocking=True),
        "loss10": label_shard["loss10"].index_select(0, idx).to(device, non_blocking=True),
        "loss20": label_shard["loss20"].index_select(0, idx).to(device, non_blocking=True),
        "unsafe_large_city": label_shard["unsafe_large_city"].index_select(0, idx).to(device, non_blocking=True),
    }


def select_player(logits: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, players.view(-1, 1)).squeeze(1)


def binary_stats(logits: torch.Tensor, target: torch.Tensor) -> dict:
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (pred == target).float().mean().item()
        positives = target.sum().item()
        return {"accuracy": accuracy, "positive_rate": positives / max(float(target.numel()), 1.0)}


def balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight_scale: float,
) -> torch.Tensor:
    positives = target.sum()
    negatives = target.numel() - positives
    if positives <= 0:
        pos_weight = torch.ones((), dtype=logits.dtype, device=logits.device)
    else:
        pos_weight = (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=20.0)
        pos_weight = pos_weight * float(pos_weight_scale)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train auxiliary city-risk heads on frozen best features.")
    parser.add_argument("--shards-dir", type=Path, default=Path("dataset/processed/imitation_shards_counterfactual_v4_residual"))
    parser.add_argument("--labels-dir", type=Path, default=Path(""))
    parser.add_argument("--agent-dir", type=Path, default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/auxiliary_risk_head_v1"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--loss10-weight", type=float, default=1.0)
    parser.add_argument("--loss20-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-large-city-weight", type=float, default=0.05)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    flags = load_agent_flags(args.agent_dir)
    model = create_model(flags, device)
    weights_path = args.agent_dir / "lux_ai" / "rl_agent" / "candidate_weights.pt"
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    aux_head = AuxiliaryRiskHead(int(model.base_out_channels)).to(device)
    optimizer = torch.optim.AdamW(aux_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    shard_paths = sorted(args.shards_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise ValueError(f"No shards found in {args.shards_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(shard_paths)
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu")
            label_shard = None
            if args.labels_dir:
                label_path = args.labels_dir / shard_path.name
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing label cache for {shard_path.name}: {label_path}")
                label_shard = torch.load(label_path, map_location="cpu")
            n = len(shard["meta"])
            for indices in iter_batches(n, args.batch_size, shuffle=True):
                model_input, meta = batch_from_shard(shard, indices, device)
                if label_shard is None:
                    labels = labels_from_meta(meta, device)
                else:
                    labels = cached_labels_for_indices(label_shard, indices, device)
                with torch.no_grad():
                    actor_features, input_mask = auxiliary_feature_forward(model, model_input)
                outputs = aux_head(actor_features.detach(), input_mask)
                loss10_logits = select_player(outputs["loss10_logit"], labels["players"])
                loss20_logits = select_player(outputs["loss20_logit"], labels["players"])
                unsafe_logits = select_player(outputs["unsafe_large_city_logit"], labels["players"])
                loss = (
                    args.loss10_weight * balanced_bce_with_logits(
                        loss10_logits,
                        labels["loss10"],
                        args.pos_weight_scale,
                    )
                    + args.loss20_weight * balanced_bce_with_logits(
                        loss20_logits,
                        labels["loss20"],
                        args.pos_weight_scale,
                    )
                    + args.unsafe_large_city_weight * F.binary_cross_entropy_with_logits(
                        unsafe_logits,
                        labels["unsafe_large_city"],
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(aux_head.parameters(), 5.0)
                optimizer.step()
                global_step += 1
                if global_step == 1 or global_step % args.log_interval == 0:
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(loss.detach().cpu()),
                        "loss10": binary_stats(loss10_logits, labels["loss10"]),
                        "loss20": binary_stats(loss20_logits, labels["loss20"]),
                        "unsafe_large_city": binary_stats(unsafe_logits, labels["unsafe_large_city"]),
                    }
                    history.append(row)
                    print(
                        "step={step} epoch={epoch} loss={loss:.4f} "
                        "loss10_acc={l10:.3f} loss20_acc={l20:.3f} unsafe_acc={ul:.3f}".format(
                            step=global_step,
                            epoch=epoch,
                            loss=row["loss"],
                            l10=row["loss10"]["accuracy"],
                            l20=row["loss20"]["accuracy"],
                            ul=row["unsafe_large_city"]["accuracy"],
                        )
                    )
                if args.max_batches and global_step >= args.max_batches:
                    break
            if args.max_batches and global_step >= args.max_batches:
                break
        if args.max_batches and global_step >= args.max_batches:
            break

    torch.save(
        {
            "aux_head_state_dict": aux_head.state_dict(),
            "in_channels": int(model.base_out_channels),
            "tasks": ["loss10", "loss20", "unsafe_large_city"],
        },
        args.output_dir / "auxiliary_risk_head.pt",
    )
    summary = {
        "agent_dir": str(args.agent_dir),
        "shards_dir": str(args.shards_dir),
        "labels_dir": str(args.labels_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss10_weight": args.loss10_weight,
        "loss20_weight": args.loss20_weight,
        "unsafe_large_city_weight": args.unsafe_large_city_weight,
        "pos_weight_scale": args.pos_weight_scale,
        "steps": global_step,
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {args.output_dir / 'auxiliary_risk_head.pt'}")
    print(f"summary: {args.output_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
