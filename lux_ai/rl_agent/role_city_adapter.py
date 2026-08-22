from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from ..lux.constants import Constants
from ..lux.game_objects import Unit
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX
from ..utility_constants import MAX_BOARD_SIZE
from .role_assignment import (
    ATTACKER,
    BUILDER,
    FIREFIGHTER,
    FUEL_DEPOT,
    FUEL_STATION,
    HARVESTER,
    MANUFACTURING_POINT,
    RESEARCH_STATION,
    SACRIFICIAL_DECAY,
    RoleAssignmentConfig,
    RoleAssignmentSnapshot,
    RoleState,
    assign_roles,
    city_tile_roles,
    direction_towards,
    nearest_city_center_by_role,
    nearest_critical_city_center,
    nearest_enemy_worker_position,
    nearest_fuel_position,
)


def pos_to_loc(pos: Tuple[int, int], board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> int:
    return pos[0] * board_dims[1] + pos[1]


class RoleCityAdapter(nn.Module):
    """Plug-in role/city adapter that can stay heuristic or learn bias strengths.

    The adapter owns role cooldown state and applies legal-mask-respecting logit
    biases after the frozen Actor produces base logits. When learnable_biases is
    enabled, the scalar bias strengths are nn.Parameters and can be optimized by
    a KL-APPO loop without changing the Actor checkpoint structure.
    """

    def __init__(self, config: RoleAssignmentConfig, learnable_biases: bool = False):
        super().__init__()
        self.config = config
        self.state = RoleState()
        self.snapshot: Optional[RoleAssignmentSnapshot] = None
        self.learnable_biases = learnable_biases
        if learnable_biases:
            self.bias_params = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(float(value), dtype=torch.float32))
                for name, value in config.bias_params.to_mapping().items()
            })
        else:
            self.bias_params = None

    @classmethod
    def from_config(cls, config: RoleAssignmentConfig) -> "RoleCityAdapter":
        return cls(config=config, learnable_biases=config.learnable_biases)

    def bias_value(self, name: str, like: torch.Tensor) -> torch.Tensor:
        if self.bias_params is not None:
            return self.bias_params[name].to(device=like.device, dtype=like.dtype)
        return torch.tensor(
            float(getattr(self.config.bias_params, name)),
            device=like.device,
            dtype=like.dtype,
        )

    def update(
            self,
            *,
            game_state,
            player,
            opponent,
            risk_blocked_positions=(),
    ) -> Optional[RoleAssignmentSnapshot]:
        if not self.config.enabled:
            self.snapshot = None
            return None
        self.snapshot = assign_roles(
            game_state=game_state,
            player=player,
            opponent=opponent,
            state=self.state,
            config=self.config,
            risk_blocked_positions=risk_blocked_positions,
        )
        return self.snapshot

    def apply(
            self,
            *,
            game_state,
            player,
            opponent,
            actionable_workers: Dict[int, List[Unit]],
            actionable_city_tiles: Dict[int, object],
            policy_logits: Dict[str, torch.Tensor],
            available_actions_mask: Dict[str, torch.Tensor],
            player_id: int,
    ) -> Dict[str, torch.Tensor]:
        if not self.config.enabled or not self.config.bias_enabled or self.snapshot is None:
            return policy_logits

        out = dict(policy_logits)
        worker_logits = policy_logits["worker"].clone()
        city_logits = policy_logits["city_tile"].clone()
        worker_mask = available_actions_mask["worker"]
        city_mask = available_actions_mask["city_tile"]

        for workers in actionable_workers.values():
            if not workers:
                continue
            worker = workers[0]
            assignment = self.snapshot.unit_roles.get(worker.id)
            if assignment is None:
                continue
            x, y = worker.pos.astuple()
            if assignment.role == HARVESTER:
                cell = game_state.map.get_cell(x, y)
                if cell.has_resource():
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "NO-OP",
                                             "harvester_mine_bias")
                target = nearest_fuel_position(game_state, player, worker.pos)
                direction = direction_towards(worker.pos, target)
                if direction is not None:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                             "harvester_move_bias")
            elif assignment.role == BUILDER:
                if worker.can_build(game_state.map):
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "BUILD_CITY",
                                             "builder_build_city_bias")
                else:
                    target = nearest_city_center_by_role(player, self.snapshot, FUEL_STATION, worker.pos)
                    direction = direction_towards(worker.pos, target)
                    if direction is not None:
                        self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                                 "builder_move_to_fuel_station_bias")
            elif assignment.role == ATTACKER:
                target = nearest_enemy_worker_position(opponent, worker.pos)
                direction = direction_towards(worker.pos, target)
                if direction is not None:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                             "attacker_block_move_bias")
                self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "BUILD_CITY",
                                         "attacker_build_city_penalty", sign=-1.0)
            elif assignment.role == FIREFIGHTER:
                target = nearest_critical_city_center(player, self.snapshot, worker.pos)
                direction = direction_towards(worker.pos, target)
                if direction is not None:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                             "firefighter_move_bias")
                self._bias_firefighter_transfers(worker_logits, worker_mask, player_id, player.units, worker, target)
                self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "BUILD_CITY",
                                         "firefighter_build_city_penalty", sign=-1.0)

        for pos, city_role in city_tile_roles(player, self.snapshot).items():
            x, y = pos
            if pos_to_loc(pos) not in actionable_city_tiles:
                continue
            if city_role == MANUFACTURING_POINT:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "BUILD_WORKER",
                                       "manufacturing_build_worker_bias")
            elif city_role == RESEARCH_STATION:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "RESEARCH",
                                       "research_station_research_bias")
            elif city_role == FUEL_DEPOT:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "RESEARCH", "depot_research_bias")
            elif city_role == FUEL_STATION:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "RESEARCH",
                                       "fuel_station_research_bias")
            elif city_role == SACRIFICIAL_DECAY:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "NO-OP", "sacrificial_noop_bias")

        out["worker"] = worker_logits
        out["city_tile"] = city_logits
        return out

    def _bias_worker_action(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            player_id: int,
            x: int,
            y: int,
            action: str,
            param_name: str,
            sign: float = 1.0,
    ) -> None:
        idx = ACTION_MEANINGS_TO_IDX["worker"][action]
        if mask[0, 0, player_id, x, y, idx]:
            logits[0, 0, player_id, x, y, idx] = (
                logits[0, 0, player_id, x, y, idx] +
                self.bias_value(param_name, logits) * sign
            )

    def _bias_city_action(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            player_id: int,
            x: int,
            y: int,
            action: str,
            param_name: str,
    ) -> None:
        idx = ACTION_MEANINGS_TO_IDX["city_tile"][action]
        if mask[0, 0, player_id, x, y, idx]:
            logits[0, 0, player_id, x, y, idx] = (
                logits[0, 0, player_id, x, y, idx] +
                self.bias_value(param_name, logits)
            )

    def _bias_firefighter_transfers(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            player_id: int,
            units,
            worker: Unit,
            target,
    ) -> None:
        if target is None:
            return
        x, y = worker.pos.astuple()
        directions = {
            Constants.DIRECTIONS.NORTH: (0, -1),
            Constants.DIRECTIONS.EAST: (1, 0),
            Constants.DIRECTIONS.SOUTH: (0, 1),
            Constants.DIRECTIONS.WEST: (-1, 0),
        }
        for resource in ("uranium", "coal", "wood"):
            if worker.cargo.get(resource) <= 0:
                continue
            for direction, (dx, dy) in directions.items():
                recipient = next((u for u in units if u.pos.x == x + dx and u.pos.y == y + dy), None)
                if recipient is None:
                    continue
                # Lux 2021 transfer is unit-to-adjacent-unit only. This only biases relay transfers
                # toward a unit that is at least as close to the critical city as the carrier.
                if recipient.pos.distance_to(target) > worker.pos.distance_to(target):
                    continue
                action = f"TRANSFER_{resource}_{direction}"
                idx = ACTION_MEANINGS_TO_IDX["worker"][action]
                if mask[0, 0, player_id, x, y, idx]:
                    logits[0, 0, player_id, x, y, idx] = (
                        logits[0, 0, player_id, x, y, idx] +
                        self.bias_value("firefighter_transfer_bias", logits)
                    )
