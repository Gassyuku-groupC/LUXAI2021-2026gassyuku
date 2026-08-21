#!/usr/bin/env python3
"""Train a frozen-best spatial residual policy head.

This is Residual Head v2: the frozen best model provides actor feature maps and
policy logits, while a tiny zero-initialized conv head learns local logit deltas.
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
from lux_ai.rl_agent.residual_head import SpatialResidualHead, action_sizes_from_logits  # noqa: E402
from train_imitation_bc import batch_from_shard, iter_batches  # noqa: E402
from train_residual_head import (  # noqa: E402
    critical_ce_loss,
    kl_anchor_loss,
    load_critical_mask,
    residual_l2_loss,
    safe_logits,
)


def frozen_actor_features(model: torch.nn.Module, model_input: dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    with torch.no_grad():
        x, input_mask, available_actions_mask, _ = model.dict_input_layer(model_input)
        base_out, _ = model.base_model((x, input_mask))
        actor_features = model.actor_base(base_out)
        policy_logits, _ = model.actor(
            actor_features,
            available_actions_mask=available_actions_mask,
            sample=False,
        )
    return actor_features.detach(), {key: value.detach() for key, value in policy_logits.items()}


def label_family_masks(shard: dict, indices: Iterable[int], device: torch.device) -> Dict[str, torch.Tensor]:
    meta = shard.get("meta") or []
    selected = [meta[int(i)] if int(i) < len(meta) else {} for i in indices]
    masks = {
        "support": [],
        "late": [],
        "safe": [],
        "penalty": [],
    }
    for row in selected:
        reason = str(row.get("weight_reason", ""))
        masks["support"].append(1.0 if "support" in reason else 0.0)
        masks["late"].append(1.0 if "late" in reason else 0.0)
        masks["safe"].append(1.0 if "safe" in reason else 0.0)
        masks["penalty"].append(1.0 if "penalty" in reason else 0.0)
    return {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in masks.items()
    }


def teacher_player_multipliers(
    shard: dict,
    indices: Iterable[int],
    device: torch.device,
    p0_mult: float,
    p1_mult: float,
) -> torch.Tensor:
    meta = shard.get("meta") or []
    values = []
    for i in indices:
        row = meta[int(i)] if int(i) < len(meta) else {}
        player = int(row.get("teacher_player", 0) or 0)
        values.append(float(p1_mult if player == 1 else p0_mult))
    return torch.tensor(values, dtype=torch.float32, device=device)


def structured_critical_ce_loss(
    final_logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: torch.Tensor,
    critical_mask: torch.Tensor,
    family_masks: Dict[str, torch.Tensor],
    anchor_weight: float,
    support_worker_mult: float,
    support_city_mult: float,
    late_worker_mult: float,
    late_city_mult: float,
    safe_city_mult: float,
) -> Tuple[torch.Tensor, dict]:
    total_loss = torch.zeros((), device=weights.device)
    total_weight = torch.zeros((), device=weights.device)
    stats = {}
    sample_weights = weights.view(-1, 1, 1, 1, 1)
    base_mask = torch.where(
        critical_mask.view(-1, 1, 1, 1, 1) > 0,
        torch.ones_like(sample_weights),
        torch.full_like(sample_weights, float(anchor_weight)),
    )
    support = family_masks["support"].view(-1, 1, 1, 1, 1)
    late = family_masks["late"].view(-1, 1, 1, 1, 1)
    safe = family_masks["safe"].view(-1, 1, 1, 1, 1)
    critical_weight_sum = torch.zeros((), device=weights.device)
    active_weight_sum = torch.zeros((), device=weights.device)

    for space, logits in final_logits.items():
        target = targets[space].float()
        action_counts = target.sum(dim=-1)
        active = action_counts > 0
        if not active.any():
            stats[f"{space}_count"] = 0.0
            continue
        if space == "worker":
            family_mult = 1.0 + support * (support_worker_mult - 1.0) + late * (late_worker_mult - 1.0)
        elif space == "city_tile":
            family_mult = 1.0 + support * (support_city_mult - 1.0) + late * (late_city_mult - 1.0) + safe * (safe_city_mult - 1.0)
        else:
            family_mult = torch.ones_like(base_mask)
        log_probs = F.log_softmax(safe_logits(logits), dim=-1)
        selected = torch.where(target > 0, log_probs, torch.zeros_like(log_probs))
        selected = torch.nan_to_num(selected, nan=0.0, neginf=-30.0, posinf=0.0)
        per_pos_loss = -selected.sum(dim=-1) / action_counts.clamp(min=1.0)
        weighted = active.float() * sample_weights * base_mask * family_mult
        space_loss = (per_pos_loss * weighted).sum()
        space_weight = weighted.sum().clamp(min=1.0)
        total_loss = total_loss + space_loss
        total_weight = total_weight + space_weight
        critical_weight_sum = critical_weight_sum + (
            active.float() * sample_weights * (critical_mask.view(-1, 1, 1, 1, 1) > 0).float()
        ).sum()
        active_weight_sum = active_weight_sum + (active.float() * sample_weights).sum()
        with torch.no_grad():
            pred = safe_logits(logits).argmax(dim=-1)
            correct = ((target.gather(-1, pred.unsqueeze(-1)).squeeze(-1) > 0) * active).sum()
            count = active.sum().clamp(min=1)
            stats[f"{space}_loss"] = (space_loss / space_weight).detach().item()
            stats[f"{space}_accuracy"] = (correct / count).detach().item()
            stats[f"{space}_count"] = active.sum().detach().item()
    stats["critical_weight_rate"] = (critical_weight_sum / active_weight_sum.clamp(min=1.0)).detach().item()
    stats["support_rate"] = family_masks["support"].mean().detach().item()
    stats["late_rate"] = family_masks["late"].mean().detach().item()
    stats["safe_rate"] = family_masks["safe"].mean().detach().item()
    return total_loss / total_weight.clamp(min=1.0), stats


def copy_spatial_residual_agent(source: Path, target: Path, overwrite: bool, residual_source: Path) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already exists; pass --overwrite-output")
        shutil.rmtree(target)
    shutil.copytree(source, target)
    rl_dir = target / "lux_ai" / "rl_agent"
    shutil.copy2(PROJECT_ROOT / "lux_ai" / "rl_agent" / "residual_head.py", rl_dir / "residual_head.py")
    shutil.copy2(residual_source, rl_dir / "spatial_residual_head.pt")
    patch_rl_agent(rl_dir / "rl_agent.py")


def patch_rl_agent(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))",
        "CHECKPOINT_PATH = Path(__file__).parent / \"candidate_weights.pt\"",
    )
    if "spatial_residual_forward" not in text:
        text = text.replace(
            "from ..nns import create_model, models\n",
            "from ..nns import create_model, models\nfrom .residual_head import SpatialResidualHead, spatial_residual_forward\n",
        )
    if "self.spatial_residual_head = None" not in text:
        marker = "        self.model.eval()\n"
        insert = (
            "        self.spatial_residual_head = None\n"
            "        spatial_residual_path = Path(__file__).parent / \"spatial_residual_head.pt\"\n"
            "        if spatial_residual_path.exists():\n"
            "            residual_state = torch.load(spatial_residual_path, map_location=\"cpu\")\n"
            "            self.spatial_residual_head = SpatialResidualHead(\n"
            "                in_channels=int(residual_state[\"in_channels\"]),\n"
            "                action_sizes=residual_state[\"action_sizes\"],\n"
            "                action_plane_shapes=residual_state[\"action_plane_shapes\"],\n"
            "                gamma=float(residual_state.get(\"gamma\", 0.05)),\n"
            "                max_delta=float(residual_state.get(\"max_delta\", 0.75)),\n"
            "                hidden_channels=int(residual_state.get(\"hidden_channels\", 64)),\n"
            "                kernel_size=int(residual_state.get(\"kernel_size\", 3)),\n"
            "                player_scales=tuple(residual_state.get(\"player_scales\", [1.0, 1.0])),\n"
            "            )\n"
            "            self.spatial_residual_head.load_state_dict(residual_state[\"model_state_dict\"])\n"
            "            self.spatial_residual_head.to(self.device)\n"
            "            self.spatial_residual_head.eval()\n"
            "\n"
        )
        text = text.replace(marker, marker + insert)
    old = "            agent_output_augmented = self.model.select_best_actions(relevant_env_output_augmented)\n"
    new = (
        "            if self.spatial_residual_head is not None:\n"
        "                agent_output_augmented = spatial_residual_forward(\n"
        "                    self.model,\n"
        "                    relevant_env_output_augmented,\n"
        "                    self.spatial_residual_head,\n"
        "                    sample=False,\n"
        "                )\n"
        "            else:\n"
        "                agent_output_augmented = self.model.select_best_actions(relevant_env_output_augmented)\n"
    )
    if old in text and new not in text:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-best spatial residual policy head.")
    parser.add_argument("--shards-dir", type=Path, default=Path("dataset/processed/imitation_shards_counterfactual_v4_residual"))
    parser.add_argument("--agent-dir", type=Path, default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"))
    parser.add_argument("--output-agent-dir", type=Path, default=Path("outputs/spatial_residual_head_v2_from_best/agent"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--max-delta", type=float, default=0.75)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--p0-residual-scale", type=float, default=1.0)
    parser.add_argument("--p1-residual-scale", type=float, default=1.0)
    parser.add_argument("--p0-ce-mult", type=float, default=1.0)
    parser.add_argument("--p1-ce-mult", type=float, default=1.0)
    parser.add_argument("--p0-anchor-mult", type=float, default=1.0)
    parser.add_argument("--p1-anchor-mult", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.2)
    parser.add_argument("--l2-beta", type=float, default=0.02)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--structured-critical-loss", action="store_true")
    parser.add_argument("--support-worker-mult", type=float, default=1.5)
    parser.add_argument("--support-city-mult", type=float, default=0.25)
    parser.add_argument("--late-worker-mult", type=float, default=0.75)
    parser.add_argument("--late-city-mult", type=float, default=1.25)
    parser.add_argument("--safe-city-mult", type=float, default=1.15)
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
    first_features, first_logits = frozen_actor_features(base_model, first_input)
    action_sizes = action_sizes_from_logits(first_logits)
    action_plane_shapes = {
        key: tuple(int(v) for v in base_model.actor.action_plane_shapes[key])
        for key in action_sizes
    }
    residual = SpatialResidualHead(
        in_channels=int(first_features.shape[1]),
        action_sizes=action_sizes,
        action_plane_shapes=action_plane_shapes,
        gamma=args.gamma,
        max_delta=args.max_delta,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        player_scales=(args.p0_residual_scale, args.p1_residual_scale),
    ).to(device)
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
                ce_weights = weights * teacher_player_multipliers(
                    shard,
                    indices,
                    device,
                    args.p0_ce_mult,
                    args.p1_ce_mult,
                )
                anchor_weights = weights * teacher_player_multipliers(
                    shard,
                    indices,
                    device,
                    args.p0_anchor_mult,
                    args.p1_anchor_mult,
                )
                actor_features, base_logits = frozen_actor_features(base_model, model_input)
                final_logits, deltas = residual(actor_features, base_logits)
                if args.structured_critical_loss:
                    family_masks = label_family_masks(shard, indices, device)
                    ce, stats = structured_critical_ce_loss(
                        final_logits,
                        targets,
                        ce_weights,
                        critical_mask,
                        family_masks,
                        args.anchor_weight,
                        args.support_worker_mult,
                        args.support_city_mult,
                        args.late_worker_mult,
                        args.late_city_mult,
                        args.safe_city_mult,
                    )
                else:
                    ce, stats = critical_ce_loss(final_logits, targets, ce_weights, critical_mask, args.anchor_weight)
                kl = kl_anchor_loss(base_logits, final_logits, targets, anchor_weights)
                l2 = residual_l2_loss(deltas, targets, anchor_weights)
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
    residual_path = out_parent / "spatial_residual_head.pt"
    torch.save({
        "model_state_dict": residual.cpu().state_dict(),
        "in_channels": int(first_features.shape[1]),
        "action_sizes": action_sizes,
        "action_plane_shapes": {key: list(value) for key, value in action_plane_shapes.items()},
        "gamma": args.gamma,
        "max_delta": args.max_delta,
        "hidden_channels": args.hidden_channels,
        "kernel_size": args.kernel_size,
        "player_scales": [args.p0_residual_scale, args.p1_residual_scale],
    }, residual_path)
    copy_spatial_residual_agent(args.agent_dir, args.output_agent_dir, args.overwrite_output, residual_path)
    summary = {
        "source_agent_dir": str(args.agent_dir),
        "output_agent_dir": str(args.output_agent_dir),
        "shards_dir": str(args.shards_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "max_delta": args.max_delta,
        "hidden_channels": args.hidden_channels,
        "kernel_size": args.kernel_size,
        "p0_residual_scale": args.p0_residual_scale,
        "p1_residual_scale": args.p1_residual_scale,
        "p0_ce_mult": args.p0_ce_mult,
        "p1_ce_mult": args.p1_ce_mult,
        "p0_anchor_mult": args.p0_anchor_mult,
        "p1_anchor_mult": args.p1_anchor_mult,
        "kl_beta": args.kl_beta,
        "l2_beta": args.l2_beta,
        "anchor_weight": args.anchor_weight,
        "structured_critical_loss": args.structured_critical_loss,
        "support_worker_mult": args.support_worker_mult,
        "support_city_mult": args.support_city_mult,
        "late_worker_mult": args.late_worker_mult,
        "late_city_mult": args.late_city_mult,
        "safe_city_mult": args.safe_city_mult,
        "steps": global_step,
        "action_sizes": action_sizes,
        "action_plane_shapes": {key: list(value) for key, value in action_plane_shapes.items()},
        "history": history,
    }
    summary_path = out_parent / "spatial_residual_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved spatial residual agent: {args.output_agent_dir}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
