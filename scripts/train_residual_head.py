#!/usr/bin/env python3
"""Train a minimal frozen-best residual policy head.

The base Lux model is frozen. Only a tiny logit-space residual head is trained:

    final_logits = best_logits + gamma * residual(best_logits)

CE is applied mainly on critical states, while KL and delta L2 keep the residual
anchored to the frozen best policy.
"""

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
from lux_ai.nns import models  # noqa: E402
from lux_ai.rl_agent.residual_head import LogitResidualHead, action_sizes_from_logits  # noqa: E402
from train_imitation_bc import batch_from_shard, iter_batches, to_device  # noqa: E402


def safe_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=-30.0, neginf=-30.0, posinf=30.0)


def masked_final_logits(base_logits: Dict[str, torch.Tensor], residual: LogitResidualHead) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    deltas = residual.delta_logits(base_logits)
    final = {}
    for key, logits in base_logits.items():
        finite_mask = torch.isfinite(logits)
        combined = safe_logits(logits) + residual.gamma * deltas[key]
        final[key] = torch.where(finite_mask, combined, logits)
    return final, deltas


def critical_ce_loss(
    final_logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: torch.Tensor,
    critical_mask: torch.Tensor,
    anchor_weight: float,
) -> Tuple[torch.Tensor, dict]:
    total_loss = torch.zeros((), device=weights.device)
    total_weight = torch.zeros((), device=weights.device)
    stats = {}
    sample_weights = weights.view(-1, 1, 1, 1, 1)
    sample_mask = torch.where(
        critical_mask.view(-1, 1, 1, 1, 1) > 0,
        torch.ones_like(sample_weights),
        torch.full_like(sample_weights, float(anchor_weight)),
    )
    critical_weight_sum = torch.zeros((), device=weights.device)
    active_weight_sum = torch.zeros((), device=weights.device)
    for space, logits in final_logits.items():
        target = targets[space].float()
        action_counts = target.sum(dim=-1)
        active = action_counts > 0
        if not active.any():
            stats[f"{space}_count"] = 0.0
            continue
        log_probs = F.log_softmax(safe_logits(logits), dim=-1)
        selected = torch.where(target > 0, log_probs, torch.zeros_like(log_probs))
        selected = torch.nan_to_num(selected, nan=0.0, neginf=-30.0, posinf=0.0)
        per_pos_loss = -selected.sum(dim=-1) / action_counts.clamp(min=1.0)
        weighted = active.float() * sample_weights * sample_mask
        space_loss = (per_pos_loss * weighted).sum()
        space_weight = weighted.sum().clamp(min=1.0)
        total_loss = total_loss + space_loss
        total_weight = total_weight + space_weight
        critical_weight_sum = critical_weight_sum + (active.float() * sample_weights * (critical_mask.view(-1, 1, 1, 1, 1) > 0).float()).sum()
        active_weight_sum = active_weight_sum + (active.float() * sample_weights).sum()
        with torch.no_grad():
            pred = safe_logits(logits).argmax(dim=-1)
            correct = ((target.gather(-1, pred.unsqueeze(-1)).squeeze(-1) > 0) * active).sum()
            count = active.sum().clamp(min=1)
            stats[f"{space}_loss"] = (space_loss / space_weight).detach().item()
            stats[f"{space}_accuracy"] = (correct / count).detach().item()
            stats[f"{space}_count"] = active.sum().detach().item()
    stats["critical_weight_rate"] = (critical_weight_sum / active_weight_sum.clamp(min=1.0)).detach().item()
    return total_loss / total_weight.clamp(min=1.0), stats


def kl_anchor_loss(
    base_logits: Dict[str, torch.Tensor],
    final_logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    total = torch.zeros((), device=weights.device)
    total_weight = torch.zeros((), device=weights.device)
    sample_weights = weights.view(-1, 1, 1, 1, 1)
    for space, base in base_logits.items():
        target = targets[space].float()
        active = target.sum(dim=-1) > 0
        if not active.any():
            continue
        base_safe = safe_logits(base.detach())
        final_safe = safe_logits(final_logits[space])
        base_prob = F.softmax(base_safe, dim=-1)
        base_log_prob = F.log_softmax(base_safe, dim=-1)
        final_log_prob = F.log_softmax(final_safe, dim=-1)
        per_pos_kl = (base_prob * (base_log_prob - final_log_prob)).sum(dim=-1)
        weighted = active.float() * sample_weights
        total = total + (per_pos_kl * weighted).sum()
        total_weight = total_weight + weighted.sum()
    return total / total_weight.clamp(min=1.0)


def residual_l2_loss(deltas: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), device=weights.device)
    total_weight = torch.zeros((), device=weights.device)
    sample_weights = weights.view(-1, 1, 1, 1, 1)
    for space, delta in deltas.items():
        active = targets[space].float().sum(dim=-1) > 0
        if not active.any():
            continue
        per_pos = delta.pow(2).mean(dim=-1)
        weighted = active.float() * sample_weights
        total = total + (per_pos * weighted).sum()
        total_weight = total_weight + weighted.sum()
    return total / total_weight.clamp(min=1.0)


def load_critical_mask(shard: dict, indices: Iterable[int], device: torch.device, min_delta: float) -> torch.Tensor:
    idx = torch.tensor(list(indices), dtype=torch.long)
    if "critical_mask" in shard:
        mask = shard["critical_mask"].index_select(0, idx).float()
    elif "counterfactual_scale" in shard:
        mask = (shard["counterfactual_scale"].index_select(0, idx).float() - 1.0).abs() >= min_delta
    else:
        mask = torch.ones(len(idx), dtype=torch.float32)
    return mask.to(device, non_blocking=True)


def copy_residual_agent(source: Path, target: Path, overwrite: bool, residual_source: Path, gamma: float, max_delta: float) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already exists; pass --overwrite-output")
        shutil.rmtree(target)
    shutil.copytree(source, target)
    rl_dir = target / "lux_ai" / "rl_agent"
    shutil.copy2(PROJECT_ROOT / "lux_ai" / "rl_agent" / "residual_head.py", rl_dir / "residual_head.py")
    shutil.copy2(residual_source, rl_dir / "residual_head.pt")
    (rl_dir / "residual_config.json").write_text(json.dumps({"gamma": gamma, "max_delta": max_delta}, indent=2), encoding="utf-8")
    patch_rl_agent(rl_dir / "rl_agent.py")


def patch_rl_agent(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))",
        "CHECKPOINT_PATH = Path(__file__).parent / \"candidate_weights.pt\"",
    )
    if "residual_head import LogitResidualHead" not in text:
        text = text.replace(
            "from ..nns import create_model, models\n",
            "from ..nns import create_model, models\nfrom .residual_head import LogitResidualHead\n",
        )
    if "self.residual_head = None" not in text:
        marker = "        self.model.eval()\n"
        insert = (
            "        self.residual_head = None\n"
            "        residual_path = Path(__file__).parent / \"residual_head.pt\"\n"
            "        if residual_path.exists():\n"
            "            residual_state = torch.load(residual_path, map_location=\"cpu\")\n"
            "            self.residual_head = LogitResidualHead(\n"
            "                residual_state[\"action_sizes\"],\n"
            "                gamma=float(residual_state.get(\"gamma\", 0.15)),\n"
            "                max_delta=float(residual_state.get(\"max_delta\", 2.0)),\n"
            "            )\n"
            "            self.residual_head.load_state_dict(residual_state[\"model_state_dict\"])\n"
            "            self.residual_head.to(self.device)\n"
            "            self.residual_head.eval()\n"
            "\n"
        )
        text = text.replace(marker, marker + insert)
    if "self.residual_head(agent_output[\"policy_logits\"])" not in text:
        marker = (
            "            agent_output[\"actions\"] = {\n"
        )
        insert = (
            "            if self.residual_head is not None:\n"
            "                agent_output[\"policy_logits\"] = {key: val.to(self.device) for key, val in agent_output[\"policy_logits\"].items()}\n"
            "                agent_output[\"policy_logits\"] = self.residual_head(agent_output[\"policy_logits\"])\n"
            "                agent_output[\"policy_logits\"] = {key: val.cpu() for key, val in agent_output[\"policy_logits\"].items()}\n"
        )
        text = text.replace(marker, insert + marker)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-best residual policy head.")
    parser.add_argument("--shards-dir", type=Path, default=Path("dataset/processed/imitation_shards_counterfactual_v4_residual"))
    parser.add_argument("--agent-dir", type=Path, default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"))
    parser.add_argument("--output-agent-dir", type=Path, default=Path("outputs/residual_head_v1_from_best/agent"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--max-delta", type=float, default=2.0)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--l2-beta", type=float, default=0.001)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--critical-min-scale-delta", type=float, default=0.02)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    flags = load_agent_flags(args.agent_dir)
    base_model = create_model(flags, device)
    checkpoint = torch.load(args.agent_dir / "lux_ai" / "rl_agent" / "candidate_weights.pt", map_location=device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    shard_paths = sorted(args.shards_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise ValueError(f"No shards found in {args.shards_dir}")

    first_shard = torch.load(shard_paths[0], map_location="cpu")
    first_input, _, _ = batch_from_shard(first_shard, [0], device)
    with torch.no_grad():
        first_logits = base_model(first_input, sample=False)["policy_logits"]
    action_sizes = action_sizes_from_logits(first_logits)
    residual = LogitResidualHead(action_sizes, gamma=args.gamma, max_delta=args.max_delta).to(device)
    optimizer = torch.optim.Adam(residual.parameters(), lr=args.lr)

    history = []
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(shard_paths)
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu")
            for indices in iter_batches(shard, args.batch_size, shuffle=True):
                model_input, targets, weights = batch_from_shard(shard, indices, device)
                critical_mask = load_critical_mask(shard, indices, device, args.critical_min_scale_delta)
                with torch.no_grad():
                    base_logits = base_model(model_input, sample=False)["policy_logits"]
                final_logits, deltas = masked_final_logits(base_logits, residual)
                ce, stats = critical_ce_loss(final_logits, targets, weights, critical_mask, args.anchor_weight)
                kl = kl_anchor_loss(base_logits, final_logits, targets, weights)
                l2 = residual_l2_loss(deltas, targets, weights)
                loss = ce + args.kl_beta * kl + args.l2_beta * l2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                if global_step % args.log_interval == 0 or global_step == 1:
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.detach().item(),
                        "ce_loss": ce.detach().item(),
                        "kl_loss": kl.detach().item(),
                        "l2_loss": l2.detach().item(),
                        **stats,
                    }
                    history.append(row)
                    print(
                        "step={step} epoch={epoch} loss={loss:.4f} ce={ce_loss:.4f} "
                        "kl={kl_loss:.5f} l2={l2_loss:.5f} critical={critical_weight_rate:.3f} "
                        f"worker_acc={row.get('worker_accuracy', 0.0):.3f} "
                        f"city_acc={row.get('city_tile_accuracy', 0.0):.3f}".format(**row)
                    )
                if args.max_batches and global_step >= args.max_batches:
                    break
            if args.max_batches and global_step >= args.max_batches:
                break
        if args.max_batches and global_step >= args.max_batches:
            break

    out_parent = args.output_agent_dir.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    residual_path = out_parent / "residual_head.pt"
    torch.save({
        "model_state_dict": residual.cpu().state_dict(),
        "action_sizes": action_sizes,
        "gamma": args.gamma,
        "max_delta": args.max_delta,
    }, residual_path)
    copy_residual_agent(args.agent_dir, args.output_agent_dir, args.overwrite_output, residual_path, args.gamma, args.max_delta)
    summary = {
        "source_agent_dir": str(args.agent_dir),
        "output_agent_dir": str(args.output_agent_dir),
        "shards_dir": str(args.shards_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "max_delta": args.max_delta,
        "kl_beta": args.kl_beta,
        "l2_beta": args.l2_beta,
        "anchor_weight": args.anchor_weight,
        "steps": global_step,
        "action_sizes": action_sizes,
        "history": history,
    }
    summary_path = out_parent / "residual_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved residual agent: {args.output_agent_dir}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
