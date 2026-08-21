#!/usr/bin/env python3
"""Jointly behavior-clone a student Actor and pooled-KV spatial sidecar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from imitation_learning_utils import load_agent_flags  # noqa: E402
from train_auxiliary_risk_head import batch_from_shard  # noqa: E402
from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX  # noqa: E402
from lux_ai.nns import create_model  # noqa: E402
from lux_ai.rl_agent.auxiliary_heads import auxiliary_feature_forward  # noqa: E402
from lux_ai.rl_agent.spatial_risk_sidecar import SpatialRiskAttentionSidecar  # noqa: E402
from lux_ai.rl_agent.learned_intervention_gate import SidecarLogitDeltaGate  # noqa: E402


def stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


@lru_cache(maxsize=128)
def replay_timeline(path_text: str) -> list[list[dict[str, list[tuple[int, int]]]]]:
    replay = json.loads(Path(path_text).read_text(encoding="utf-8"))
    timeline = []
    for step in replay.get("steps") or []:
        by_team = [{}, {}]
        updates = step[0].get("observation", {}).get("updates") or []
        for update in updates:
            parts = update.split()
            if len(parts) >= 6 and parts[0] == "ct":
                team, city_id, x, y = int(parts[1]), parts[2], int(parts[3]), int(parts[4])
                by_team[team].setdefault(city_id, []).append((x, y))
        timeline.append(by_team)
    return timeline


def spatial_targets(shard: dict, indices: list[int], device: torch.device):
    batch = len(indices)
    risk = torch.zeros(batch, 2, 32, 32, device=device)
    risk_mask = torch.zeros_like(risk, dtype=torch.bool)
    safe = torch.zeros_like(risk)
    safe_mask = torch.zeros_like(risk, dtype=torch.bool)
    build_city_idx = ACTION_MEANINGS_TO_IDX["worker"]["BUILD_CITY"]
    selected_actions = shard["actions_taken"]["worker"].index_select(
        0, torch.tensor(indices, dtype=torch.long)
    )[..., build_city_idx]
    while selected_actions.dim() > 4:
        selected_actions = selected_actions.any(dim=1)

    for local_index, shard_index in enumerate(indices):
        meta = shard["meta"][shard_index]
        player = int(meta["teacher_player"])
        step = int(meta["state_step"])
        timeline = replay_timeline(str(meta["file"]))
        current = timeline[min(step, len(timeline) - 1)][player]
        future20 = timeline[min(step + 20, len(timeline) - 1)][player]
        future40 = timeline[min(step + 40, len(timeline) - 1)][player]
        for city_id, coordinates in current.items():
            shrinks = len(future20.get(city_id, [])) < len(coordinates)
            for x, y in coordinates:
                risk_mask[local_index, player, x, y] = True
                risk[local_index, player, x, y] = float(shrinks)
        candidate = selected_actions[local_index, player].bool()
        if candidate.any():
            current_count = sum(len(value) for value in current.values())
            future20_count = sum(len(value) for value in future20.values())
            future40_count = sum(len(value) for value in future40.values())
            risk[local_index, player][candidate] = float(current_count - future20_count >= 5)
            risk_mask[local_index, player][candidate] = True
            safe[local_index, player][candidate] = float(future40_count > current_count)
            safe_mask[local_index, player][candidate] = True
    return risk, risk_mask, safe, safe_mask


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    selected_targets = targets[mask]
    positives = selected_targets.sum().clamp(min=1.0)
    negatives = (selected_targets.numel() - selected_targets.sum()).clamp(min=1.0)
    pos_weight = (negatives / positives).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(logits[mask], selected_targets, pos_weight=pos_weight)


def behavior_cloning_loss(policy_logits: dict, shard: dict, indices: list[int], device: torch.device):
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    for action_space, logits in policy_logits.items():
        target = shard["actions_taken"][action_space].index_select(0, index_tensor).to(device)
        available = shard["available_actions_mask"][action_space].index_select(
            0, index_tensor
        ).to(device=device, dtype=torch.bool)
        selected = (target.bool() & available).to(dtype=logits.dtype)
        # Illegal actions are represented by -inf. Multiplying their log-probability
        # by a zero one-hot target produces NaN, even though they are not selected.
        finite_logits = torch.where(
            torch.isfinite(logits), logits, torch.full_like(logits, -1e4)
        )
        total = total - (F.log_softmax(finite_logits, dim=-1) * selected).sum()
        count = count + selected.sum()
    return total / count.clamp(min=1.0)


def invalid_target_count(shard: dict, indices: list[int], device: torch.device) -> int:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    invalid = 0
    for action_space, target in shard["actions_taken"].items():
        selected = target.index_select(0, index_tensor).to(device=device, dtype=torch.bool)
        available = shard["available_actions_mask"][action_space].index_select(
            0, index_tensor
        ).to(device=device, dtype=torch.bool)
        invalid += int((selected & ~available).sum().item())
    return invalid


def split_indices(shard: dict, wanted_sizes: set[int], validation_fraction: float):
    """Split by replay/episode identity so adjacent frames never cross splits."""
    result = {"train": [], "validation": []}
    for index, meta in enumerate(shard["meta"]):
        map_size = int(meta.get("width", meta.get("map_size", 0)) or 0)
        if map_size not in wanted_sizes:
            continue
        group = str(meta.get("episode_id") or meta.get("seed") or meta.get("file"))
        split = "validation" if stable_fraction(group) < validation_fraction else "train"
        result[split].append(index)
    return result


def composite_state_dict(actor, sidecar, gate):
    state = {f"base_agent.{key}": value for key, value in actor.state_dict().items()}
    state.update({f"spatial_risk_sidecar.{key}": value for key, value in sidecar.state_dict().items()})
    state.update({f"intervention_gate.{key}": value for key, value in gate.state_dict().items()})
    return state


METRIC_FIELDS = (
    "epoch", "shard", "shards", "batch", "step", "samples", "samples_per_second",
    "loss", "risk_loss", "safe_loss", "bc_loss", "grad_norm", "actor_lr",
    "sidecar_lr", "gpu_memory_gb",
    "invalid_target_actions",
)


def append_metric(output_dir: Path, metric: dict) -> None:
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(metric, allow_nan=False) + "\n")
    csv_path = output_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: metric.get(key) for key in METRIC_FIELDS})


def save_recovery(output_dir: Path, actor, sidecar, gate, optimizer, epoch: int, step: int) -> None:
    torch.save({
        "model_state_dict": composite_state_dict(actor, sidecar, gate),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }, output_dir / "recovery.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--student-agent-dir", type=Path, default=Path("internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research"))
    parser.add_argument("--student-checkpoint", default="062179520_weights.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/first_actor_sidecar_bc"))
    parser.add_argument("--map-sizes", default="12,16,24,32")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sidecar-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)
    wanted_sizes = {int(value) for value in args.map_sizes.split(",") if value.strip()}
    flags = load_agent_flags(args.student_agent_dir)
    actor = create_model(flags, device)
    checkpoint_path = args.student_agent_dir / "lux_ai" / "rl_agent" / args.student_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.train()
    sidecar = SpatialRiskAttentionSidecar(actor.base_out_channels).to(device)
    gate = SidecarLogitDeltaGate().to(device)
    optimizer = torch.optim.AdamW([
        {"params": actor.parameters(), "lr": args.actor_lr},
        {"params": list(sidecar.parameters()) + list(gate.parameters()), "lr": args.sidecar_lr},
    ], weight_decay=1e-4)
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=device)
        state = resume["model_state_dict"]
        actor.load_state_dict({
            key.removeprefix("base_agent."): value
            for key, value in state.items() if key.startswith("base_agent.")
        }, strict=True)
        sidecar.load_state_dict({
            key.removeprefix("spatial_risk_sidecar."): value
            for key, value in state.items() if key.startswith("spatial_risk_sidecar.")
        }, strict=True)
        gate.load_state_dict({
            key.removeprefix("intervention_gate."): value
            for key, value in state.items() if key.startswith("intervention_gate.")
        }, strict=True)
        if resume.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        print(f"Resumed joint BC checkpoint: {args.resume_checkpoint}", flush=True)
    shards = sorted(args.shards_dir.glob("shard_*.pt"))
    history = []
    global_step = 0
    best_validation = float("inf")
    stale_epochs = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    samples_seen = 0
    invalid_targets_seen = 0
    print(
        f"BC start device={device} shards={len(shards)} maps={sorted(wanted_sizes)} "
        f"epochs<={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )
    for epoch in range(args.epochs):
        random.shuffle(shards)
        epoch_metrics = {"train": [], "validation": []}
        for shard_number, shard_path in enumerate(shards, start=1):
            shard = torch.load(shard_path, map_location="cpu")
            split = split_indices(shard, wanted_sizes, args.validation_fraction)
            train_indices = split["train"]
            random.shuffle(train_indices)
            for start in range(0, len(train_indices), args.batch_size):
                indices = train_indices[start:start + args.batch_size]
                if not indices:
                    continue
                model_input, _ = batch_from_shard(shard, indices, device)
                actor_features, input_mask = auxiliary_feature_forward(actor, model_input)
                base_output = actor.select_best_actions(model_input)
                outputs = sidecar(actor_features.detach(), input_mask)
                risk, risk_mask, safe, safe_mask = spatial_targets(shard, indices, device)
                risk_loss = masked_bce(outputs["risk_logits"], risk, risk_mask)
                safe_loss = masked_bce(outputs["safe_expansion_logits"], safe, safe_mask)
                gated_logits, _ = gate(
                    base_output["policy_logits"],
                    outputs["risk_logits"],
                    safe_expansion_logits=outputs["safe_expansion_logits"],
                )
                bc_loss = behavior_cloning_loss(gated_logits, shard, indices, device)
                invalid_targets_seen += invalid_target_count(shard, indices, device)
                loss = risk_loss + safe_loss + bc_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite BC loss at step={global_step + 1} shard={shard_path}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(sidecar.parameters()) + list(gate.parameters()), 5.0
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite BC gradient at step={global_step + 1} shard={shard_path}"
                    )
                optimizer.step()
                global_step += 1
                samples_seen += len(indices)
                history.append({"step": global_step, "loss": float(loss.detach()), "risk": float(risk_loss.detach()), "safe": float(safe_loss.detach()), "bc": float(bc_loss.detach())})
                epoch_metrics["train"].append(float(loss.detach()))
                if global_step == 1 or global_step % args.log_every == 0:
                    elapsed = max(time.perf_counter() - run_started, 1e-6)
                    gpu_memory = (
                        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        if device.type == "cuda" else 0.0
                    )
                    metric = {
                        "epoch": epoch + 1,
                        "shard": shard_number,
                        "shards": len(shards),
                        "batch": start // args.batch_size + 1,
                        "step": global_step,
                        "samples": samples_seen,
                        "samples_per_second": samples_seen / elapsed,
                        "loss": float(loss.detach()),
                        "risk_loss": float(risk_loss.detach()),
                        "safe_loss": float(safe_loss.detach()),
                        "bc_loss": float(bc_loss.detach()),
                        "grad_norm": float(grad_norm.detach()),
                        "actor_lr": optimizer.param_groups[0]["lr"],
                        "sidecar_lr": optimizer.param_groups[1]["lr"],
                        "gpu_memory_gb": gpu_memory,
                        "invalid_target_actions": invalid_targets_seen,
                    }
                    append_metric(args.output_dir, metric)
                    print(
                        "BC "
                        f"epoch={metric['epoch']}/{args.epochs} "
                        f"shard={shard_number}/{len(shards)} step={global_step} "
                        f"loss={metric['loss']:.4f} bc={metric['bc_loss']:.4f} "
                        f"risk={metric['risk_loss']:.4f} safe={metric['safe_loss']:.4f} "
                        f"grad={metric['grad_norm']:.3f} samples/s={metric['samples_per_second']:.1f} "
                        f"gpu={gpu_memory:.2f}GB invalid_targets={invalid_targets_seen}",
                        flush=True,
                    )
                if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                    save_recovery(args.output_dir, actor, sidecar, gate, optimizer, epoch + 1, global_step)
                if args.max_batches and global_step >= args.max_batches:
                    break
            if args.max_batches and global_step >= args.max_batches:
                break
            actor.eval()
            sidecar.eval()
            gate.eval()
            with torch.no_grad():
                for start in range(0, len(split["validation"]), args.batch_size):
                    indices = split["validation"][start:start + args.batch_size]
                    if not indices:
                        continue
                    model_input, _ = batch_from_shard(shard, indices, device)
                    actor_features, input_mask = auxiliary_feature_forward(actor, model_input)
                    base_output = actor.select_best_actions(model_input)
                    outputs = sidecar(actor_features.detach(), input_mask)
                    risk, risk_mask, safe, safe_mask = spatial_targets(shard, indices, device)
                    gated_logits, _ = gate(base_output["policy_logits"], outputs["risk_logits"], safe_expansion_logits=outputs["safe_expansion_logits"])
                    val_loss = masked_bce(outputs["risk_logits"], risk, risk_mask) + masked_bce(outputs["safe_expansion_logits"], safe, safe_mask) + behavior_cloning_loss(gated_logits, shard, indices, device)
                    epoch_metrics["validation"].append(float(val_loss))
            actor.train()
            sidecar.train()
            gate.train()
        if args.max_batches and global_step >= args.max_batches:
            break
        train_mean = sum(epoch_metrics["train"]) / max(1, len(epoch_metrics["train"]))
        validation_mean = sum(epoch_metrics["validation"]) / max(1, len(epoch_metrics["validation"]))
        if not math.isfinite(train_mean) or not math.isfinite(validation_mean):
            raise FloatingPointError(
                f"Non-finite epoch metrics: train={train_mean}, validation={validation_mean}"
            )
        history.append({"epoch": epoch + 1, "train_mean": train_mean, "validation_mean": validation_mean})
        print(
            f"BC epoch_complete={epoch + 1} train={train_mean:.6f} "
            f"validation={validation_mean:.6f} best={best_validation:.6f}",
            flush=True,
        )
        if validation_mean < best_validation - args.min_delta:
            best_validation = validation_mean
            stale_epochs = 0
            torch.save({"model_state_dict": composite_state_dict(actor, sidecar, gate), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch + 1, "validation_loss": validation_mean, "student_checkpoint": str(checkpoint_path)}, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    latest_checkpoint = {
        "model_state_dict": composite_state_dict(actor, sidecar, gate),
        "spatial_risk_sidecar_state_dict": sidecar.state_dict(),
        "intervention_gate_state_dict": gate.state_dict(),
        "base_model_state_dict": actor.state_dict(),
        "student_checkpoint": str(checkpoint_path),
        "map_sizes": sorted(wanted_sizes),
        "epochs_completed": epoch + 1,
    }
    torch.save(latest_checkpoint, args.output_dir / "latest.pt")
    if not (args.output_dir / "best.pt").exists():
        torch.save(latest_checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "train_summary.json").write_text(json.dumps({"steps": global_step, "best_validation_loss": best_validation, "history": history[-100:]}, indent=2), encoding="utf-8")
    print(f"steps={global_step} best={args.output_dir / 'best.pt'} latest={args.output_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
