"""Core simulation logic for movement, combat, deaths, and episode resets."""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import (
    Action,
    ATTACK_ACTIONS,
    ITEM_ACTIONS,
    MOVEMENT_ACTIONS,
    SHOOT_ACTIONS,
    action_to_delta,
)
from .config import (
    LOW_HEALTH_THRESHOLD,
    MAP_HEIGHT,
    MAP_WIDTH,
    MAX_EPISODE_LENGTH,
    MAX_HEALTH,
    NUM_AGENTS,
    UNARMED_DAMAGE,
    VISION_RADIUS,
)
from .entity import AgentEntity, ItemEntity
from .items import HealItem, MeleeWeaponItem, RangedWeaponItem, create_item_entity
from .mapgen import GridMap, generate_map
from .observation import compute_visible_cells
from .utils import create_id_counter, in_bounds


@dataclass(slots=True)
class DamageEvent:
    """Describes a single successful melee or ranged hit."""

    attacker_id: int
    target_id: int
    damage: int
    target_position: tuple[int, int]
    direction: str
    weapon_type: str | None = None
    used_weapon: bool = False
    killed_target: bool = False


@dataclass(slots=True)
class StepResult:
    """Summarizes the outcome of one world simulation step."""

    tick: int
    episode_over: bool
    winner_id: int | None
    deaths: list[int] = field(default_factory=list)
    damage_events: list[DamageEvent] = field(default_factory=list)
    item_events: list[dict[str, int | str]] = field(default_factory=list)


class World:
    """Owns the arena map, agents, and deterministic Phase 1 state updates."""

    def __init__(
        self,
        width: int = MAP_WIDTH,
        height: int = MAP_HEIGHT,
        num_agents: int = NUM_AGENTS,
        max_health: int = MAX_HEALTH,
        melee_damage: int = UNARMED_DAMAGE,
        max_episode_length: int = MAX_EPISODE_LENGTH,
    ) -> None:
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.max_health = max_health
        self.melee_damage = melee_damage
        self.max_episode_length = max_episode_length

        self._id_counter = create_id_counter()
        self.tick = 0
        self.grid_map: GridMap = generate_map(width, height)
        self.agents: dict[int, AgentEntity] = {}
        self.items: dict[int, ItemEntity] = {}
        self._occupancy: dict[tuple[int, int], int] = {}
        self._item_occupancy: dict[tuple[int, int], int] = {}
        self._visible_cells_cache: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self._nearest_visible_enemy_distance_cache: dict[int, int | None] = {}
        self._shot_target_cache: dict[tuple[int, Action], AgentEntity | None] = {}
        self.spawn_points = self._build_spawn_points()
        self.reset()

    def reset(self) -> None:
        """Reset the world state and respawn all agents."""

        self.tick = 0
        self.grid_map = generate_map(self.width, self.height)
        self._clear_step_caches()
        if not self.agents:
            self.agents = self._create_agents()
            self.items = self._create_items()
            self._rebuild_occupancy()
            self._rebuild_item_occupancy()
            return
        for agent, spawn_point in zip(self.agents.values(), self.spawn_points, strict=False):
            agent.spawn_x, agent.spawn_y = spawn_point
            agent.reset_for_episode()
        self.items = self._create_items()
        self._rebuild_occupancy()
        self._rebuild_item_occupancy()

    def step(self, agent_actions: dict[int, Action]) -> StepResult:
        """Advance the world by one tick using the provided agent actions."""

        self.tick += 1
        self._clear_step_caches()
        previously_alive = {agent.entity_id for agent in self.alive_agents()}
        for agent in self.agents.values():
            agent.clear_step_flags()
            agent.previous_visible_enemy_distance = self._nearest_visible_enemy_distance(agent)
            agent.current_visible_enemy_distance = agent.previous_visible_enemy_distance

        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            if not agent.alive:
                continue
            action = agent_actions.get(agent_id, Action.WAIT)
            agent.last_action = action
            if action in MOVEMENT_ACTIONS:
                self.move_agent(agent, action)

        self._clear_step_caches()
        damage_events: list[DamageEvent] = []
        item_events: list[dict[str, int | str]] = []
        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            if not agent.alive:
                continue
            action = agent_actions.get(agent_id, Action.WAIT)
            if action in ITEM_ACTIONS:
                event = self.handle_item_action(agent, action)
                if event is not None:
                    item_events.append(event)

        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            if not agent.alive:
                continue
            action = agent_actions.get(agent_id, Action.WAIT)
            if action in ATTACK_ACTIONS:
                event = self.perform_attack(agent, action)
                if event is not None:
                    damage_events.append(event)
            elif action in SHOOT_ACTIONS:
                event = self.perform_shot(agent, action)
                if event is not None:
                    damage_events.append(event)

        self._clear_step_caches()
        for agent_id in previously_alive:
            agent = self.agents[agent_id]
            agent.current_visible_enemy_distance = self._nearest_visible_enemy_distance(agent)
            agent.entered_attack_range_this_step = (
                not self._is_attack_range_distance(agent.previous_visible_enemy_distance)
                and self._is_attack_range_distance(agent.current_visible_enemy_distance)
            )
            self.agents[agent_id].record_position()

        deaths = [
            agent.entity_id
            for agent in self.agents.values()
            if agent.entity_id in previously_alive and not agent.alive
        ]
        episode_over = self.is_episode_over()
        winner_id = self.get_winner_id()
        return StepResult(
            tick=self.tick,
            episode_over=episode_over,
            winner_id=winner_id,
            deaths=deaths,
            damage_events=damage_events,
            item_events=item_events,
        )

    def move_agent(self, agent: AgentEntity, action: Action) -> bool:
        """Attempt to move an agent one tile and return whether it succeeded."""

        dx, dy = action_to_delta(action)
        target_x = agent.x + dx
        target_y = agent.y + dy
        if self.is_blocked(target_x, target_y):
            return False
        self._occupancy.pop((agent.x, agent.y), None)
        agent.x = target_x
        agent.y = target_y
        self._occupancy[(agent.x, agent.y)] = agent.entity_id
        self._clear_step_caches()
        return True

    def perform_attack(self, agent: AgentEntity, action: Action) -> DamageEvent | None:
        """Attack the adjacent target cell and return a damage event on hit."""

        dx, dy = action_to_delta(action)
        target_x = agent.x + dx
        target_y = agent.y + dy
        target = self.get_agent_at(target_x, target_y)
        if target is None or target.entity_id == agent.entity_id or not target.alive:
            return None

        used_weapon = agent.has_active_weapon_bonus()
        weapon_type = agent.equipped_weapon_type if used_weapon else None
        applied_damage = agent.consume_melee_attack_damage(self.melee_damage)
        target.apply_damage(applied_damage, source_direction=action.name)
        agent.damage_dealt += applied_damage
        killed_target = not target.alive
        if killed_target:
            agent.kills += 1
        self._clear_step_caches()

        return DamageEvent(
            attacker_id=agent.entity_id,
            target_id=target.entity_id,
            damage=applied_damage,
            target_position=(target_x, target_y),
            direction=action.name,
            weapon_type=weapon_type,
            used_weapon=used_weapon,
            killed_target=killed_target,
        )

    def perform_shot(self, agent: AgentEntity, action: Action) -> DamageEvent | None:
        """Fire a ranged shot in one direction until the first wall or target is hit."""

        if not agent.has_ranged_weapon():
            return None

        dx, dy = action_to_delta(action)
        max_range = max(1, agent.equipped_weapon_range)
        target = None
        target_position = None

        for distance in range(1, max_range + 1):
            target_x = agent.x + dx * distance
            target_y = agent.y + dy * distance
            if not in_bounds(target_x, target_y, self.width, self.height):
                break
            if self.grid_map.is_wall(target_x, target_y):
                break
            occupant = self.get_agent_at(target_x, target_y)
            if occupant is not None and occupant.entity_id != agent.entity_id:
                target = occupant
                target_position = (target_x, target_y)
                break

        if target is None or target_position is None:
            return None

        applied_damage = agent.consume_ranged_attack_damage(self.melee_damage)
        target.apply_damage(applied_damage, source_direction=action.name)
        agent.damage_dealt += applied_damage
        killed_target = not target.alive
        if killed_target:
            agent.kills += 1
        self._clear_step_caches()

        return DamageEvent(
            attacker_id=agent.entity_id,
            target_id=target.entity_id,
            damage=applied_damage,
            target_position=target_position,
            direction=action.name,
            weapon_type="ranged_weapon",
            used_weapon=True,
            killed_target=killed_target,
        )

    def handle_item_action(self, agent: AgentEntity, action: Action) -> dict[str, int | str] | None:
        """Resolve pickup or use-item actions and return a compact event record."""

        visible_enemy_distance = self._nearest_visible_enemy_distance(agent)
        visible_enemy_at_event = int(visible_enemy_distance is not None)
        low_health_at_event = int(agent.health / max(1, agent.max_health) <= LOW_HEALTH_THRESHOLD)

        if action == Action.PICKUP:
            item = self.get_item_at(agent.x, agent.y)
            if item is None or agent.has_inventory_item():
                return None
            if item.item_type in {"melee_weapon", "ranged_weapon"} and agent.has_equipped_weapon():
                return None
            reward_granted = int(agent.entity_id not in item.pickup_rewarded_agent_ids)
            if reward_granted:
                item.pickup_rewarded_agent_ids.add(agent.entity_id)
            if not agent.store_item(item):
                return None
            self.items.pop(item.entity_id, None)
            self._item_occupancy.pop((item.x, item.y), None)
            self._clear_step_caches()
            return {
                "agent_id": agent.entity_id,
                "item_type": item.item_type,
                "event_type": "pickup",
                "from_drop": int(item.spawned_by_drop),
                "reward_granted": reward_granted,
                "visible_enemy_at_event": visible_enemy_at_event,
                "low_health_at_event": low_health_at_event,
                "value": 1,
            }

        if action == Action.USE_ITEM:
            effect = agent.use_inventory_item()
            if effect is None:
                return None
            if effect[0] == "weapon":
                effect_type, effect_value, from_drop = effect
                return {
                    "agent_id": agent.entity_id,
                    "item_type": effect_type,
                    "event_type": "use",
                    "from_drop": int(from_drop),
                    "value": effect_value,
                }
            effect_type, effect_value = effect
            return {
                "agent_id": agent.entity_id,
                "item_type": effect_type,
                "event_type": "use",
                "value": effect_value,
            }

        if action == Action.DROP_ITEM:
            if self.get_item_at(agent.x, agent.y) is not None:
                return None
            dropped = agent.drop_item_or_weapon()
            if dropped is None:
                return None
            (
                item_type,
                charges,
                heal_amount,
                damage_bonus,
                item_range,
                _item_from_drop,
                reward_lineage_id,
                rewarded_agent_ids,
            ) = dropped
            item_id = next(self._id_counter)
            dropped_item = ItemEntity(
                entity_id=item_id,
                x=agent.x,
                y=agent.y,
                item_type=str(item_type),
                charges=int(charges),
                heal_amount=int(heal_amount),
                damage_bonus=int(damage_bonus),
                item_range=int(item_range),
                spawned_by_drop=True,
                reward_lineage_id=int(reward_lineage_id),
                pickup_rewarded_agent_ids=set(rewarded_agent_ids),
            )
            self.items[item_id] = dropped_item
            self._item_occupancy[(agent.x, agent.y)] = item_id
            self._clear_step_caches()
            return {
                "agent_id": agent.entity_id,
                "item_type": str(item_type),
                "event_type": "drop",
                "visible_enemy_at_event": visible_enemy_at_event,
                "value": int(charges),
            }

        return None

    def get_agent_at(self, x: int, y: int) -> AgentEntity | None:
        """Return the living agent occupying the given coordinate."""

        agent_id = self._occupancy.get((x, y))
        if agent_id is None:
            return None
        agent = self.agents[agent_id]
        return agent if agent.alive else None

    def get_item_at(self, x: int, y: int) -> ItemEntity | None:
        """Return the item occupying the given coordinate, if one exists."""

        item_id = self._item_occupancy.get((x, y))
        if item_id is None:
            return None
        item = self.items[item_id]
        return item if item.alive else None

    def is_blocked(self, x: int, y: int) -> bool:
        """Return True when movement into the target cell is not allowed."""

        if not in_bounds(x, y, self.width, self.height):
            return True
        if self.grid_map.is_wall(x, y):
            return True
        return self.get_agent_at(x, y) is not None

    def alive_agents(self) -> list[AgentEntity]:
        """Return all currently living agents."""

        return [agent for agent in self.agents.values() if agent.alive]

    def get_legal_actions(self, agent: AgentEntity) -> list[Action]:
        """Return actions that are currently legal for the given agent."""

        legal_actions = [Action.WAIT]
        for action in MOVEMENT_ACTIONS:
            dx, dy = action_to_delta(action)
            if not self.is_blocked(agent.x + dx, agent.y + dy):
                legal_actions.append(action)
        for action in ATTACK_ACTIONS:
            dx, dy = action_to_delta(action)
            target = self.get_agent_at(agent.x + dx, agent.y + dy)
            if target is not None and target.entity_id != agent.entity_id:
                legal_actions.append(action)
        if self.get_item_at(agent.x, agent.y) is not None and not agent.has_inventory_item():
            item = self.get_item_at(agent.x, agent.y)
            if item is not None and (
                item.item_type not in {"melee_weapon", "ranged_weapon"}
                or not agent.has_equipped_weapon()
            ):
                legal_actions.append(Action.PICKUP)
        if agent.has_inventory_item() and (
            (
                agent.inventory_item_type == "heal"
                and agent.health < agent.max_health
            )
            or (
                agent.inventory_item_type in {"melee_weapon", "ranged_weapon"}
                and not agent.has_equipped_weapon()
            )
        ):
            legal_actions.append(Action.USE_ITEM)
        if (
            self.get_item_at(agent.x, agent.y) is None
            and (agent.has_inventory_item() or agent.has_equipped_weapon())
        ):
            legal_actions.append(Action.DROP_ITEM)
        if agent.has_ranged_weapon():
            for action, target in self._get_shot_targets(agent).items():
                if target is not None:
                    legal_actions.append(action)
        return legal_actions

    def get_visible_cells(self, agent: AgentEntity, radius: int) -> set[tuple[int, int]]:
        """Return cached visible cells for the current stable world state."""

        cache_key = (agent.entity_id, radius)
        cached_cells = self._visible_cells_cache.get(cache_key)
        if cached_cells is not None:
            return cached_cells
        visible_cells = compute_visible_cells(self, agent, radius)
        self._visible_cells_cache[cache_key] = visible_cells
        return visible_cells

    def is_episode_over(self) -> bool:
        """Return True when the episode should terminate."""

        return len(self.alive_agents()) <= 1 or self.tick >= self.max_episode_length

    def get_winner_id(self) -> int | None:
        """Return the surviving agent identifier, or None on draw."""

        living = self.alive_agents()
        if len(living) == 1:
            return living[0].entity_id
        return None

    def _create_agents(self) -> dict[int, AgentEntity]:
        """Create all initial agent entities for the arena."""

        agents: dict[int, AgentEntity] = {}
        for spawn_x, spawn_y in self.spawn_points:
            agent_id = next(self._id_counter)
            agents[agent_id] = AgentEntity(
                entity_id=agent_id,
                x=spawn_x,
                y=spawn_y,
                spawn_x=spawn_x,
                spawn_y=spawn_y,
                max_health=self.max_health,
                health=self.max_health,
            )
        return agents

    def _create_items(self) -> dict[int, ItemEntity]:
        """Create deterministic heal and weapon pickups for the current episode."""

        candidate_blueprints = [
            ((2, self.height // 2), HealItem()),
            ((self.width - 3, self.height // 2), HealItem()),
            ((self.width // 2, 2), MeleeWeaponItem()),
            ((self.width // 2, self.height - 3), MeleeWeaponItem()),
            ((2, 2), RangedWeaponItem()),
            ((self.width - 3, self.height - 3), RangedWeaponItem()),
        ]

        items: dict[int, ItemEntity] = {}
        blocked_positions = set(self.spawn_points)
        for (x, y), blueprint in candidate_blueprints:
            if (
                not in_bounds(x, y, self.width, self.height)
                or self.grid_map.is_wall(x, y)
                or (x, y) in blocked_positions
            ):
                continue
            item_id = next(self._id_counter)
            items[item_id] = create_item_entity(item_id, x, y, blueprint)
        return items

    def _rebuild_occupancy(self) -> None:
        """Rebuild the coordinate-to-agent lookup table from current living agents."""

        self._occupancy = {
            (agent.x, agent.y): agent.entity_id
            for agent in self.agents.values()
            if agent.alive
        }

    def _rebuild_item_occupancy(self) -> None:
        """Rebuild the coordinate-to-item lookup table from current world items."""

        self._item_occupancy = {
            (item.x, item.y): item.entity_id
            for item in self.items.values()
            if item.alive
        }

    def _build_spawn_points(self) -> list[tuple[int, int]]:
        """Create deterministic spawn points away from walls and obstacles."""

        candidate_points = [
            (1, 1),
            (self.width - 2, 1),
            (1, self.height - 2),
            (self.width - 2, self.height - 2),
        ]
        valid_points = [
            point for point in candidate_points if not self.grid_map.is_wall(point[0], point[1])
        ]
        return valid_points[: self.num_agents]

    def _nearest_visible_enemy_distance(self, agent: AgentEntity) -> int | None:
        """Return the nearest visible enemy distance for reward shaping."""

        if not agent.alive:
            return None
        if agent.entity_id in self._nearest_visible_enemy_distance_cache:
            return self._nearest_visible_enemy_distance_cache[agent.entity_id]

        visible_cells = self.get_visible_cells(agent, VISION_RADIUS)
        visible_distances = [
            abs(other.x - agent.x) + abs(other.y - agent.y)
            for other in self.alive_agents()
            if other.entity_id != agent.entity_id and (other.x, other.y) in visible_cells
        ]
        if not visible_distances:
            self._nearest_visible_enemy_distance_cache[agent.entity_id] = None
            return None
        nearest_distance = min(visible_distances)
        self._nearest_visible_enemy_distance_cache[agent.entity_id] = nearest_distance
        return nearest_distance

    def _find_shot_target(self, agent: AgentEntity, action: Action) -> AgentEntity | None:
        """Return the first target that would be hit by a ranged shot."""

        if not agent.has_ranged_weapon():
            return None
        cache_key = (agent.entity_id, action)
        if cache_key in self._shot_target_cache:
            return self._shot_target_cache[cache_key]

        dx, dy = action_to_delta(action)
        for distance in range(1, max(1, agent.equipped_weapon_range) + 1):
            target_x = agent.x + dx * distance
            target_y = agent.y + dy * distance
            if not in_bounds(target_x, target_y, self.width, self.height):
                self._shot_target_cache[cache_key] = None
                return None
            if self.grid_map.is_wall(target_x, target_y):
                self._shot_target_cache[cache_key] = None
                return None
            target = self.get_agent_at(target_x, target_y)
            if target is not None and target.entity_id != agent.entity_id:
                self._shot_target_cache[cache_key] = target
                return target
        self._shot_target_cache[cache_key] = None
        return None

    def _get_shot_targets(self, agent: AgentEntity) -> dict[Action, AgentEntity | None]:
        """Return one cached shot-target lookup for each shoot direction."""

        return {
            action: self._find_shot_target(agent, action)
            for action in SHOOT_ACTIONS
        }

    def _clear_step_caches(self) -> None:
        """Clear lightweight per-state caches after any world-state change."""

        self._visible_cells_cache.clear()
        self._nearest_visible_enemy_distance_cache.clear()
        self._shot_target_cache.clear()

    @staticmethod
    def _is_attack_range_distance(distance: int | None) -> bool:
        """Return True when the enemy distance is inside melee range."""

        return distance == 1
