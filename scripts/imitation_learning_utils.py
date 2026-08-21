#!/usr/bin/env python3
"""Shared utilities for replay-based imitation learning."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple
import types
import warnings

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays is extremely slow.*")

gym_notices = types.ModuleType("gym_notices")
gym_notices_notices = types.ModuleType("gym_notices.notices")
gym_notices_notices.notices = {}
gym_notices.notices = gym_notices_notices
sys.modules.setdefault("gym_notices", gym_notices)
sys.modules.setdefault("gym_notices.notices", gym_notices_notices)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from lux_ai.lux_gym import LuxEnv, wrappers
from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS, ACTION_MEANINGS_TO_IDX
from lux_ai.utils import flags_to_namespace


def load_agent_flags(agent_dir: Path) -> SimpleNamespace:
    with (agent_dir / "lux_ai" / "rl_agent" / "config.yaml").open(encoding="utf-8") as config_file:
        return flags_to_namespace(yaml.safe_load(config_file))


def build_manual_env(flags: SimpleNamespace, first_updates: List[str]):
    env = LuxEnv(
        act_space=flags.act_space(),
        obs_space=flags.obs_space(),
        configuration={},
        run_game_automatically=False,
    )
    env = env.obs_space.wrap_env(env)
    env = wrappers.PadFixedShapeEnv(env)
    env = wrappers.VecEnv([env])
    env = wrappers.PytorchEnv(env, torch.device("cpu"))
    env = wrappers.DictEnv(env)
    env.reset(observation_updates=first_updates, force=True)
    return env


def action_placeholder(env) -> Dict[str, torch.Tensor]:
    return {
        key: torch.zeros((1,) + space.shape, dtype=torch.long)
        for key, space in env.unwrapped[0].action_space.get_action_space().spaces.items()
    }


def env_output_for_current_state(env, placeholder: Dict[str, torch.Tensor]) -> Dict:
    return env.step(placeholder)


def advance_manual_env(env, step: int, updates: List[str]) -> None:
    base_env = env.unwrapped[0]
    base_env.manual_step(updates)
    base_env.game_state.turn = step
    base_env.game_state.id = 0


def empty_actions_taken(board_dims: Tuple[int, int] = (32, 32)) -> Dict[str, np.ndarray]:
    return {
        key: np.zeros((1, 2, board_dims[0], board_dims[1], len(meanings)), dtype=bool)
        for key, meanings in ACTION_MEANINGS.items()
    }


def _direction_from_delta(dx: int, dy: int) -> str:
    if dx == 0 and dy == -1:
        return "n"
    if dx == 0 and dy == 1:
        return "s"
    if dx == 1 and dy == 0:
        return "e"
    if dx == -1 and dy == 0:
        return "w"
    raise ValueError(f"Unsupported transfer delta: dx={dx} dy={dy}")


def _unit_lookup(game_state, player: int):
    by_id = {}
    by_pos = {}
    for unit in game_state.players[player].units:
        by_id[unit.id] = unit
        by_pos[(unit.pos.x, unit.pos.y)] = unit
    return by_id, by_pos


def teacher_actions_to_mask(game_state, teacher_player: int, actions: Iterable[str]) -> Dict[str, np.ndarray]:
    target = empty_actions_taken()
    unit_by_id, unit_by_pos = _unit_lookup(game_state, teacher_player)
    worker_counts = np.zeros((32, 32), dtype=int)
    cart_counts = np.zeros((32, 32), dtype=int)

    for raw_action in actions:
        parts = raw_action.split()
        if not parts:
            continue
        kind = parts[0]
        try:
            if kind == "m" and len(parts) >= 3:
                unit = unit_by_id.get(parts[1])
                if unit is None:
                    continue
                unit_type = "worker" if unit.is_worker() else "cart"
                action_name = f"MOVE_{parts[2]}"
                target[unit_type][0, teacher_player, unit.pos.x, unit.pos.y, ACTION_MEANINGS_TO_IDX[unit_type][action_name]] = True
            elif kind == "bcity" and len(parts) >= 2:
                unit = unit_by_id.get(parts[1])
                if unit is None or not unit.is_worker():
                    continue
                target["worker"][0, teacher_player, unit.pos.x, unit.pos.y, ACTION_MEANINGS_TO_IDX["worker"]["BUILD_CITY"]] = True
            elif kind in {"bw", "bc", "r"} and len(parts) >= 3:
                x, y = int(parts[1]), int(parts[2])
                action_name = {"bw": "BUILD_WORKER", "bc": "BUILD_CART", "r": "RESEARCH"}[kind]
                target["city_tile"][0, teacher_player, x, y, ACTION_MEANINGS_TO_IDX["city_tile"][action_name]] = True
            elif kind == "p" and len(parts) >= 2:
                unit = unit_by_id.get(parts[1])
                if unit is None or not unit.is_worker():
                    continue
                target["worker"][0, teacher_player, unit.pos.x, unit.pos.y, ACTION_MEANINGS_TO_IDX["worker"]["PILLAGE"]] = True
            elif kind == "t" and len(parts) >= 5:
                src = unit_by_id.get(parts[1])
                dst = unit_by_id.get(parts[2]) or unit_by_pos.get((getattr(src, "pos", None).x, getattr(src, "pos", None).y))
                if src is None or dst is None:
                    continue
                unit_type = "worker" if src.is_worker() else "cart"
                dx = dst.pos.x - src.pos.x
                dy = dst.pos.y - src.pos.y
                direction = _direction_from_delta(dx, dy)
                resource = parts[3]
                action_name = f"TRANSFER_{resource}_{direction}"
                target[unit_type][0, teacher_player, src.pos.x, src.pos.y, ACTION_MEANINGS_TO_IDX[unit_type][action_name]] = True
        except Exception:
            continue
    return target
