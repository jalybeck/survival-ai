"""Entity models for agents and later item placeholders in the world."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .actions import Action


@dataclass(slots=True)
class Entity:
    """Base world entity with position and blocking flags."""

    entity_id: int
    x: int
    y: int
    alive: bool = True
    blocks_movement: bool = False
    blocks_vision: bool = False


@dataclass(slots=True)
class AgentEntity(Entity):
    """Represents a living combatant inside the grid world."""

    spawn_x: int = 0
    spawn_y: int = 0
    max_health: int = 10
    health: int = 10
    last_action: Action = Action.WAIT
    kills: int = 0
    damage_dealt: int = 0
    damage_taken: int = 0
    damaged_last_step: bool = False
    last_damage_direction: str | None = None
    last_reward: float = 0.0
    total_reward: float = 0.0
    visited_tiles: set[tuple[int, int]] = field(default_factory=set)
    position_history: list[tuple[int, int]] = field(default_factory=list)
    visited_new_tile_this_step: bool = False
    idled_last_step: bool = False
    oscillated_last_step: bool = False
    previous_visible_enemy_distance: int | None = None
    current_visible_enemy_distance: int | None = None
    entered_attack_range_this_step: bool = False
    inventory_item_type: str | None = None
    inventory_charges: int = 0
    inventory_heal_amount: int = 0
    inventory_damage_bonus: int = 0
    inventory_range: int = 0
    inventory_from_drop: bool = False
    inventory_reward_lineage_id: int = 0
    inventory_rewarded_agent_ids: set[int] = field(default_factory=set)
    equipped_weapon_type: str | None = None
    equipped_weapon_charges: int = 0
    equipped_weapon_damage_bonus: int = 0
    equipped_weapon_range: int = 0
    equipped_weapon_reward_lineage_id: int = 0
    equipped_weapon_rewarded_agent_ids: set[int] = field(default_factory=set)
    picked_up_item_this_step: bool = False
    used_heal_item_this_step: bool = False
    used_weapon_item_this_step: bool = False
    healed_amount_this_step: int = 0
    health_before_heal_this_step: int = 0
    last_inspected_kind: str | None = None
    last_inspected_item_type: str | None = None
    last_inspected_direction: str | None = None
    last_inspected_agent_health_norm: float = 0.0
    last_inspected_agent_has_melee: bool = False
    last_inspected_agent_has_ranged: bool = False
    last_inspection_age: int = 0
    blocks_movement: bool = True

    def __post_init__(self) -> None:
        """Normalize default spawn coordinates after creation."""

        if self.spawn_x == 0 and self.spawn_y == 0:
            self.spawn_x = self.x
            self.spawn_y = self.y
        if not self.visited_tiles:
            self.visited_tiles = {(self.x, self.y)}
        if not self.position_history:
            self.position_history = [(self.x, self.y)]

    def reset_for_episode(self) -> None:
        """Restore the agent to its spawn position and full health."""

        self.x = self.spawn_x
        self.y = self.spawn_y
        self.alive = True
        self.health = self.max_health
        self.last_action = Action.WAIT
        self.kills = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.damaged_last_step = False
        self.last_damage_direction = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.visited_tiles = {(self.x, self.y)}
        self.position_history = [(self.x, self.y)]
        self.visited_new_tile_this_step = False
        self.idled_last_step = False
        self.oscillated_last_step = False
        self.previous_visible_enemy_distance = None
        self.current_visible_enemy_distance = None
        self.entered_attack_range_this_step = False
        self.inventory_item_type = None
        self.inventory_charges = 0
        self.inventory_heal_amount = 0
        self.inventory_damage_bonus = 0
        self.inventory_range = 0
        self.inventory_from_drop = False
        self.inventory_reward_lineage_id = 0
        self.inventory_rewarded_agent_ids = set()
        self.equipped_weapon_type = None
        self.equipped_weapon_charges = 0
        self.equipped_weapon_damage_bonus = 0
        self.equipped_weapon_range = 0
        self.equipped_weapon_reward_lineage_id = 0
        self.equipped_weapon_rewarded_agent_ids = set()
        self.picked_up_item_this_step = False
        self.used_heal_item_this_step = False
        self.used_weapon_item_this_step = False
        self.healed_amount_this_step = 0
        self.health_before_heal_this_step = 0
        self.last_inspected_kind = None
        self.last_inspected_item_type = None
        self.last_inspected_direction = None
        self.last_inspected_agent_health_norm = 0.0
        self.last_inspected_agent_has_melee = False
        self.last_inspected_agent_has_ranged = False
        self.last_inspection_age = 0

    def apply_damage(self, amount: int, source_direction: str | None = None) -> int:
        """Apply incoming damage and return the remaining health."""

        self.health = max(0, self.health - amount)
        self.damage_taken += amount
        self.damaged_last_step = amount > 0
        self.last_damage_direction = source_direction
        if self.health <= 0:
            self.alive = False
        return self.health

    def clear_step_flags(self) -> None:
        """Reset per-step observation and reward-shaping flags before a world tick."""

        self.damaged_last_step = False
        self.last_damage_direction = None
        self.visited_new_tile_this_step = False
        self.idled_last_step = False
        self.oscillated_last_step = False
        self.entered_attack_range_this_step = False
        self.picked_up_item_this_step = False
        self.used_heal_item_this_step = False
        self.used_weapon_item_this_step = False
        self.healed_amount_this_step = 0
        self.health_before_heal_this_step = 0

    def age_inspection_memory(self) -> None:
        """Age the latest inspection snapshot and clear it when it expires."""

        if self.last_inspection_age <= 0:
            return
        self.last_inspection_age -= 1
        if self.last_inspection_age <= 0:
            self.clear_inspection_memory()

    def clear_inspection_memory(self) -> None:
        """Remove any stored short-lived inspection result."""

        self.last_inspected_kind = None
        self.last_inspected_item_type = None
        self.last_inspected_direction = None
        self.last_inspected_agent_health_norm = 0.0
        self.last_inspected_agent_has_melee = False
        self.last_inspected_agent_has_ranged = False
        self.last_inspection_age = 0

    def record_inspection(
        self,
        *,
        kind: str,
        direction: str,
        item_type: str | None = None,
        agent_health_norm: float = 0.0,
        agent_has_melee: bool = False,
        agent_has_ranged: bool = False,
    ) -> None:
        """Store one short-lived inspection result for later observation features."""

        self.last_inspected_kind = kind
        self.last_inspected_item_type = item_type
        self.last_inspected_direction = direction
        self.last_inspected_agent_health_norm = agent_health_norm
        self.last_inspected_agent_has_melee = agent_has_melee
        self.last_inspected_agent_has_ranged = agent_has_ranged
        self.last_inspection_age = config.INSPECT_MEMORY_TICKS

    def has_inventory_item(self) -> bool:
        """Return True when the agent is carrying an item in its single-slot inventory."""

        return self.inventory_item_type is not None

    def has_active_weapon_bonus(self) -> bool:
        """Return True when the agent has an equipped melee weapon."""

        return self.equipped_weapon_type == "melee_weapon" and self.equipped_weapon_charges > 0

    def has_equipped_weapon(self) -> bool:
        """Return True when the agent currently has any active weapon equipped."""

        return self.equipped_weapon_type is not None and self.equipped_weapon_charges > 0

    def has_ranged_weapon(self) -> bool:
        """Return True when the agent currently has a ranged weapon equipped."""

        return self.equipped_weapon_type == "ranged_weapon" and self.equipped_weapon_charges > 0

    def store_item(self, item) -> bool:
        """Copy a world item into the agent inventory when the slot is empty."""

        if self.has_inventory_item():
            return False
        if item.item_type in {"melee_weapon", "ranged_weapon"} and self.has_equipped_weapon():
            return False
        self.inventory_item_type = item.item_type
        self.inventory_charges = item.charges
        self.inventory_heal_amount = item.heal_amount
        self.inventory_damage_bonus = item.damage_bonus
        self.inventory_range = item.item_range
        self.inventory_from_drop = item.spawned_by_drop
        self.inventory_reward_lineage_id = item.reward_lineage_id
        self.inventory_rewarded_agent_ids = set(item.pickup_rewarded_agent_ids)
        self.picked_up_item_this_step = True
        return True

    def use_inventory_item(self) -> tuple[str, int] | tuple[str, int, int] | None:
        """Use the carried item and return a compact effect summary."""

        if not self.has_inventory_item():
            return None

        if self.inventory_item_type == "heal":
            missing_health = self.max_health - self.health
            if missing_health <= 0:
                return None
            self.health_before_heal_this_step = self.health
            healed_amount = min(missing_health, self.inventory_heal_amount)
            self.health += healed_amount
            self.inventory_item_type = None
            self.inventory_charges = 0
            self.inventory_heal_amount = 0
            self.inventory_damage_bonus = 0
            self.inventory_range = 0
            self.inventory_from_drop = False
            self.inventory_reward_lineage_id = 0
            self.inventory_rewarded_agent_ids = set()
            self.used_heal_item_this_step = healed_amount > 0
            self.healed_amount_this_step = healed_amount
            return ("heal", healed_amount)

        if self.inventory_item_type in {"melee_weapon", "ranged_weapon"}:
            if self.has_equipped_weapon():
                return None
            self.equipped_weapon_type = self.inventory_item_type
            self.equipped_weapon_damage_bonus = self.inventory_damage_bonus
            self.equipped_weapon_charges = max(1, self.inventory_charges)
            self.equipped_weapon_range = self.inventory_range
            self.equipped_weapon_reward_lineage_id = self.inventory_reward_lineage_id
            self.equipped_weapon_rewarded_agent_ids = set(self.inventory_rewarded_agent_ids)
            from_drop = self.inventory_from_drop
            self.inventory_item_type = None
            self.inventory_charges = 0
            self.inventory_heal_amount = 0
            self.inventory_damage_bonus = 0
            self.inventory_range = 0
            self.inventory_from_drop = False
            self.inventory_reward_lineage_id = 0
            self.inventory_rewarded_agent_ids = set()
            self.used_weapon_item_this_step = True
            return ("weapon", self.equipped_weapon_charges, int(from_drop))

        return None

    def drop_item_or_weapon(
        self,
    ) -> tuple[str | None, int, int, int, int, bool, int, set[int]] | None:
        """Remove the carried item or equipped weapon and return its item stats."""

        if self.has_inventory_item():
            dropped = (
                self.inventory_item_type,
                self.inventory_charges,
                self.inventory_heal_amount,
                self.inventory_damage_bonus,
                self.inventory_range,
                self.inventory_from_drop,
                self.inventory_reward_lineage_id,
                set(self.inventory_rewarded_agent_ids),
            )
            self.inventory_item_type = None
            self.inventory_charges = 0
            self.inventory_heal_amount = 0
            self.inventory_damage_bonus = 0
            self.inventory_range = 0
            self.inventory_from_drop = False
            self.inventory_reward_lineage_id = 0
            self.inventory_rewarded_agent_ids = set()
            return dropped

        if self.has_equipped_weapon():
            dropped = (
                self.equipped_weapon_type,
                self.equipped_weapon_charges,
                0,
                self.equipped_weapon_damage_bonus,
                self.equipped_weapon_range,
                True,
                self.equipped_weapon_reward_lineage_id,
                set(self.equipped_weapon_rewarded_agent_ids),
            )
            self.equipped_weapon_type = None
            self.equipped_weapon_charges = 0
            self.equipped_weapon_damage_bonus = 0
            self.equipped_weapon_range = 0
            self.equipped_weapon_reward_lineage_id = 0
            self.equipped_weapon_rewarded_agent_ids = set()
            return dropped

        return None

    def consume_melee_attack_damage(self, base_damage: int) -> int:
        """Return the current melee damage and spend one melee weapon charge if equipped."""

        total_damage = base_damage
        if self.has_active_weapon_bonus():
            total_damage = self.equipped_weapon_damage_bonus
            self.equipped_weapon_charges -= 1
            if self.equipped_weapon_charges <= 0:
                self.equipped_weapon_type = None
                self.equipped_weapon_charges = 0
                self.equipped_weapon_damage_bonus = 0
                self.equipped_weapon_range = 0
        return total_damage

    def consume_ranged_attack_damage(self, base_damage: int) -> int:
        """Return the current ranged damage and spend one ranged weapon charge if equipped."""

        total_damage = base_damage
        if self.has_ranged_weapon():
            total_damage = self.equipped_weapon_damage_bonus
            self.equipped_weapon_charges -= 1
            if self.equipped_weapon_charges <= 0:
                self.equipped_weapon_type = None
                self.equipped_weapon_charges = 0
                self.equipped_weapon_damage_bonus = 0
                self.equipped_weapon_range = 0
        return total_damage

    def record_position(self) -> None:
        """Record the current tile and derive short-horizon movement-pattern flags."""

        current_position = (self.x, self.y)
        previous_position = self.position_history[-1] if self.position_history else None
        position_two_steps_ago = (
            self.position_history[-2]
            if len(self.position_history) >= 2
            else None
        )

        self.visited_new_tile_this_step = current_position not in self.visited_tiles
        if self.visited_new_tile_this_step:
            self.visited_tiles.add(current_position)

        self.idled_last_step = previous_position == current_position

        # A-B-A movement is a simple signature for a two-tile dance loop.
        self.oscillated_last_step = (
            position_two_steps_ago == current_position
            and previous_position is not None
            and previous_position != current_position
        )

        self.position_history.append(current_position)
        if len(self.position_history) > 6:
            self.position_history = self.position_history[-6:]


@dataclass(slots=True)
class ItemEntity(Entity):
    """Represents a world pickup that can be carried and later consumed or activated."""

    item_type: str = "generic"
    charges: int = 1
    heal_amount: int = 0
    damage_bonus: int = 0
    item_range: int = 0
    spawned_by_drop: bool = False
    reward_lineage_id: int = 0
    pickup_rewarded_agent_ids: set[int] = field(default_factory=set)
    blocks_movement: bool = False
    blocks_vision: bool = False
