#!/usr/bin/env python3
"""Behavior-cloning fine-tune from extracted imitation shards."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imitation_learning_utils import load_agent_flags  # noqa: E402
from lux_ai.nns import create_model  # noqa: E402


def to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True)


def iter_batches(shard: dict, batch_size: int, shuffle: bool):
    n = int(shard["weights"].shape[0])
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start: start + batch_size]


def batch_from_shard(shard: dict, indices: Iterable[int], device: torch.device) -> Tuple[dict, dict, torch.Tensor]:
    idx = torch.tensor(list(indices), dtype=torch.long)
    obs = {key: value.index_select(0, idx) for key, value in shard["obs"].items()}
    available = {
        key: value.index_select(0, idx) for key, value in shard["available_actions_mask"].items()
    }
    actions_taken = {
        key: value.index_select(0, idx) for key, value in shard["actions_taken"].items()
    }
    weights = shard["weights"].index_select(0, idx)
    model_input = {
        "obs": to_device(obs, device),
        "info": {
            "input_mask": shard["input_mask"].index_select(0, idx).to(device, non_blocking=True),
            "available_actions_mask": to_device(available, device),
        },
    }
    return model_input, to_device(actions_taken, device), weights.to(device, non_blocking=True)


def bc_loss(policy_logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], weights: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    total_loss = torch.zeros((), device=weights.device)
    total_weight = torch.zeros((), device=weights.device)
    stats = {}
    sample_weights = weights.view(-1, 1, 1, 1, 1)
    for space, logits in policy_logits.items():
        target = targets[space].float()
        action_counts = target.sum(dim=-1)
        active = action_counts > 0
        if not active.any():
            stats[f"{space}_count"] = 0.0
            continue
        safe_logits = torch.nan_to_num(logits, nan=-30.0, neginf=-30.0, posinf=30.0)
        log_probs = F.log_softmax(safe_logits, dim=-1)
        selected_log_probs = torch.where(target > 0, log_probs, torch.zeros_like(log_probs))
        selected_log_probs = torch.nan_to_num(selected_log_probs, nan=0.0, neginf=-30.0, posinf=0.0)
        per_pos_loss = -selected_log_probs.sum(dim=-1) / action_counts.clamp(min=1.0)
        weighted = active.float() * sample_weights
        space_loss = (per_pos_loss * weighted).sum()
        space_weight = weighted.sum().clamp(min=1.0)
        total_loss = total_loss + space_loss
        total_weight = total_weight + space_weight
        with torch.no_grad():
            pred = safe_logits.argmax(dim=-1)
            correct = ((target.gather(-1, pred.unsqueeze(-1)).squeeze(-1) > 0) * active).sum()
            count = active.sum().clamp(min=1)
            stats[f"{space}_loss"] = (space_loss / space_weight).detach().item()
            stats[f"{space}_accuracy"] = (correct / count).detach().item()
            stats[f"{space}_count"] = active.sum().detach().item()
    return total_loss / total_weight.clamp(min=1.0), stats


def copy_agent_dir(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already exists; pass --overwrite-output")
        shutil.rmtree(target)
    shutil.copytree(source, target)


def configure_trainable_parameters(model: torch.nn.Module, scope: str):
    if scope == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif scope == "actor":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("actor_base.") or name.startswith("actor.")
    else:
        raise ValueError(f"Unknown trainable scope: {scope}")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters selected for scope={scope}")
    return trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an agent with offline replay BC.")
    parser.add_argument("--shards-dir", type=Path, default=Path("dataset/processed/imitation_shards_hq"))
    parser.add_argument("--agent-dir", type=Path, default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"))
    parser.add_argument("--output-agent-dir", type=Path, default=Path("outputs/imitation_bc_hq_v1/agent"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument(
        "--trainable-scope",
        choices=["all", "actor"],
        default="all",
        help="Use actor for a conservative BC update that freezes the shared base and value head.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite-output", action="store_true")
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
    model.train()
    trainable_parameters = configure_trainable_parameters(model, args.trainable_scope)
    trainable_count = sum(param.numel() for param in trainable_parameters)
    total_count = sum(param.numel() for param in model.parameters())
    print(f"trainable_scope={args.trainable_scope} trainable_params={trainable_count}/{total_count}")

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    shard_paths = sorted(args.shards_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise ValueError(f"No shards found in {args.shards_dir}")

    history = []
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(shard_paths)
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu")
            for indices in iter_batches(shard, args.batch_size, shuffle=True):
                model_input, targets, weights = batch_from_shard(shard, indices, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(model_input, sample=False)
                loss, stats = bc_loss(outputs["policy_logits"], targets, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                global_step += 1
                if global_step % args.log_interval == 0 or global_step == 1:
                    row = {"step": global_step, "epoch": epoch, "loss": loss.detach().item(), **stats}
                    history.append(row)
                    print(
                        "step={step} epoch={epoch} loss={loss:.4f} "
                        f"worker_acc={row.get('worker_accuracy', 0.0):.3f} "
                        f"city_acc={row.get('city_tile_accuracy', 0.0):.3f}".format(**row)
                    )
                if args.max_batches and global_step >= args.max_batches:
                    break
            if args.max_batches and global_step >= args.max_batches:
                break
        if args.max_batches and global_step >= args.max_batches:
            break

    copy_agent_dir(args.agent_dir, args.output_agent_dir, args.overwrite_output)
    out_weights = args.output_agent_dir / "lux_ai" / "rl_agent" / "candidate_weights.pt"
    torch.save({"model_state_dict": model.state_dict()}, out_weights)
    summary = {
        "source_agent_dir": str(args.agent_dir),
        "output_agent_dir": str(args.output_agent_dir),
        "shards_dir": str(args.shards_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "trainable_scope": args.trainable_scope,
        "steps": global_step,
        "history": history,
    }
    summary_path = args.output_agent_dir.parent / "bc_train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved agent: {args.output_agent_dir}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
