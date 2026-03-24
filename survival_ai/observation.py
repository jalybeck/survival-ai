"""Observation helpers for visibility, local grids, and numeric feature vectors."""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .actions import SHOOT_ACTIONS
from .utils import in_bounds, manhattan_distance

UNKNOWN_CELL = "?"
EMPTY_CELL = "."
WALL_CELL = "#"
SELF_CELL = "S"
AGENT_CELL = "A"
HEAL_ITEM_CELL = "H"
MELEE_WEAPON_CELL = "M"
RANGED_WEAPON_CELL = "R"


@dataclass(slots=True)
class SelfState:
    """Stores self-only information that should always be available to the agent."""

    x_norm: float
    y_norm: float
    health_norm: float
    damaged_last_step: float
    damage_from_up: float
    damage_from_down: float
    damage_from_left: float
    damage_from_right: float


@dataclass(slots=True)
class ObservationSnapshot:
    """Packages all observation views for one agent at one point in time."""

    agent_id: int
    visible_cells: set[tuple[int, int]]
    local_grid: list[list[str]]
    local_grid_lines: list[str]
    self_state: SelfState
    feature_names: list[str]
    feature_vector: list[float]
    nearest_visible_agent_id: int | None
    nearest_visible_agent_dx: int | None
    nearest_visible_agent_dy: int | None
    nearest_visible_agent_distance: int | None


def line_of_sight(world, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Return True when a straight line between two cells is not blocked."""

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    current_x, current_y = x1, y1

    while (current_x, current_y) != (x2, y2):
        if (current_x, current_y) != (x1, y1) and world.grid_map.is_wall(current_x, current_y):
            return False
        error_twice = 2 * err
        if error_twice > -dy:
            err -= dy
            current_x += sx
        if error_twice < dx:
            err += dx
            current_y += sy

    return True


def compute_visible_cells(world, agent, radius: int) -> set[tuple[int, int]]:
    """Return all visible coordinates for an agent within the given radius."""

    visible_cells: set[tuple[int, int]] = {(agent.x, agent.y)}

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            target_x = agent.x + dx
            target_y = agent.y + dy
            if not in_bounds(target_x, target_y, world.width, world.height):
                continue
            if max(abs(dx), abs(dy)) > radius:
                continue
            if line_of_sight(world, agent.x, agent.y, target_x, target_y):
                visible_cells.add((target_x, target_y))

    return visible_cells


def build_local_grid(world, agent, radius: int, visible_cells: set[tuple[int, int]] | None = None) -> list[list[str]]:
    """Build a symbolic local grid where unseen tiles remain explicitly unknown."""

    if visible_cells is None:
        visible_cells = compute_visible_cells(world, agent, radius)

    local_grid: list[list[str]] = []
    for dy in range(-radius, radius + 1):
        row: list[str] = []
        for dx in range(-radius, radius + 1):
            world_x = agent.x + dx
            world_y = agent.y + dy
            if not in_bounds(world_x, world_y, world.width, world.height):
                row.append(UNKNOWN_CELL)
                continue
            if (world_x, world_y) not in visible_cells:
                row.append(UNKNOWN_CELL)
                continue
            row.append(_classify_visible_cell(world, agent, world_x, world_y))
        local_grid.append(row)

    return local_grid


def build_feature_vector(world, agent, radius: int) -> tuple[list[str], list[float]]:
    """Convert one agent observation into a small, inspectable numeric feature vector."""

    visible_cells = _get_visible_cells(world, agent, radius)
    return build_feature_vector_from_visible(world, agent, radius, visible_cells)


def build_feature_vector_from_visible(
    world,
    agent,
    radius: int,
    visible_cells: set[tuple[int, int]],
) -> tuple[list[str], list[float]]:
    """Build the feature vector when the caller already has visible cells available."""

    visible_agents = _visible_other_agents(world, agent, visible_cells)
    nearest_agent = _nearest_visible_agent(agent, visible_agents)
    visible_items = _visible_items(world, visible_cells)
    nearest_item = _nearest_visible_item(agent, visible_items)
    self_state = build_self_state(world, agent)

    feature_names: list[str] = []
    feature_vector: list[float] = []

    # These self-state features are always available to the policy and do not
    # depend on line of sight. Keeping them grouped makes the feature vector
    # easier to inspect when you later connect it to the custom network.
    _append_feature(feature_names, feature_vector, "self_x_norm", self_state.x_norm)
    _append_feature(feature_names, feature_vector, "self_y_norm", self_state.y_norm)
    _append_feature(feature_names, feature_vector, "self_health_norm", self_state.health_norm)

    for name, dx, dy in (
        ("wall_up", 0, -1),
        ("wall_down", 0, 1),
        ("wall_left", -1, 0),
        ("wall_right", 1, 0),
    ):
        target_x = agent.x + dx
        target_y = agent.y + dy
        _append_feature(
            feature_names,
            feature_vector,
            name,
            1.0 if world.grid_map.is_inside(target_x, target_y) and world.grid_map.is_wall(target_x, target_y) else 0.0,
        )

    for name, dx, dy in (
        ("visible_agent_up", 0, -1),
        ("visible_agent_down", 0, 1),
        ("visible_agent_left", -1, 0),
        ("visible_agent_right", 1, 0),
    ):
        target = world.get_agent_at(agent.x + dx, agent.y + dy)
        is_visible_target = (
            target is not None
            and target.entity_id != agent.entity_id
            and (target.x, target.y) in visible_cells
        )
        _append_feature(feature_names, feature_vector, name, 1.0 if is_visible_target else 0.0)

    if nearest_agent is None:
        nearest_dx_norm = 0.0
        nearest_dy_norm = 0.0
        nearest_distance_norm = 1.0
    else:
        nearest_dx_norm = _normalize_relative_axis(nearest_agent.x - agent.x, world.width)
        nearest_dy_norm = _normalize_relative_axis(nearest_agent.y - agent.y, world.height)
        nearest_distance_norm = manhattan_distance(
            (agent.x, agent.y),
            (nearest_agent.x, nearest_agent.y),
        ) / max(1, radius * 2)

    _append_feature(feature_names, feature_vector, "nearest_agent_dx_norm", nearest_dx_norm)
    _append_feature(feature_names, feature_vector, "nearest_agent_dy_norm", nearest_dy_norm)
    _append_feature(feature_names, feature_vector, "nearest_agent_distance_norm", nearest_distance_norm)
    _append_feature(feature_names, feature_vector, "visible_enemy_exists", 1.0 if nearest_agent is not None else 0.0)

    if nearest_item is None:
        nearest_item_dx_norm = 0.0
        nearest_item_dy_norm = 0.0
        nearest_item_distance_norm = 1.0
    else:
        nearest_item_dx_norm = _normalize_relative_axis(nearest_item.x - agent.x, world.width)
        nearest_item_dy_norm = _normalize_relative_axis(nearest_item.y - agent.y, world.height)
        nearest_item_distance_norm = manhattan_distance(
            (agent.x, agent.y),
            (nearest_item.x, nearest_item.y),
        ) / max(1, radius * 2)

    current_tile_item = world.get_item_at(agent.x, agent.y)
    _append_feature(feature_names, feature_vector, "item_here", 1.0 if current_tile_item is not None else 0.0)
    _append_feature(
        feature_names,
        feature_vector,
        "standing_on_heal_item",
        1.0 if current_tile_item is not None and current_tile_item.item_type == "heal" else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "standing_on_melee_weapon_item",
        1.0 if current_tile_item is not None and current_tile_item.item_type == "melee_weapon" else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "standing_on_ranged_weapon_item",
        1.0 if current_tile_item is not None and current_tile_item.item_type == "ranged_weapon" else 0.0,
    )
    _append_feature(feature_names, feature_vector, "has_item", 1.0 if agent.has_inventory_item() else 0.0)
    _append_feature(feature_names, feature_vector, "has_melee_weapon", 1.0 if agent.has_active_weapon_bonus() else 0.0)
    _append_feature(feature_names, feature_vector, "has_ranged_weapon", 1.0 if agent.has_ranged_weapon() else 0.0)
    _append_feature(
        feature_names,
        feature_vector,
        "equipped_melee_weapon",
        1.0 if agent.equipped_weapon_type == "melee_weapon" and agent.equipped_weapon_charges > 0 else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "equipped_ranged_weapon",
        1.0 if agent.equipped_weapon_type == "ranged_weapon" and agent.equipped_weapon_charges > 0 else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "equipped_weapon_charges_norm",
        _normalize_equipped_weapon_charges(agent),
    )
    _append_feature(
        feature_names,
        feature_vector,
        "carrying_heal_item",
        1.0 if agent.inventory_item_type == "heal" else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "carrying_melee_weapon_item",
        1.0 if agent.inventory_item_type == "melee_weapon" else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "carrying_ranged_weapon_item",
        1.0 if agent.inventory_item_type == "ranged_weapon" else 0.0,
    )
    _append_feature(feature_names, feature_vector, "nearest_item_dx_norm", nearest_item_dx_norm)
    _append_feature(feature_names, feature_vector, "nearest_item_dy_norm", nearest_item_dy_norm)
    _append_feature(feature_names, feature_vector, "nearest_item_distance_norm", nearest_item_distance_norm)
    _append_feature(
        feature_names,
        feature_vector,
        "visible_item_count_norm",
        len(visible_items) / max(1, len(world.items)),
    )

    inspect_known = 1.0 if agent.last_inspection_age > 0 and agent.last_inspected_kind is not None else 0.0
    inspect_direction = agent.last_inspected_direction
    inspect_kind = agent.last_inspected_kind
    inspect_item_type = agent.last_inspected_item_type
    _append_feature(feature_names, feature_vector, "inspect_known", inspect_known)
    _append_feature(
        feature_names,
        feature_vector,
        "inspect_is_current_tile",
        1.0 if inspect_direction == "SELF" and inspect_known else 0.0,
    )
    for direction_name in ("UP", "DOWN", "LEFT", "RIGHT"):
        _append_feature(
            feature_names,
            feature_vector,
            f"inspect_dir_{direction_name.lower()}",
            1.0 if inspect_direction == direction_name and inspect_known else 0.0,
        )
    for kind_name in ("empty", "wall", "item", "agent"):
        _append_feature(
            feature_names,
            feature_vector,
            f"inspect_found_{kind_name}",
            1.0 if inspect_kind == kind_name and inspect_known else 0.0,
        )
    for item_type in ("heal", "melee_weapon", "ranged_weapon"):
        suffix = (
            "heal"
            if item_type == "heal"
            else "melee" if item_type == "melee_weapon" else "ranged"
        )
        _append_feature(
            feature_names,
            feature_vector,
            f"inspect_item_{suffix}",
            1.0 if inspect_item_type == item_type and inspect_known else 0.0,
        )
    _append_feature(
        feature_names,
        feature_vector,
        "inspect_agent_health_norm",
        agent.last_inspected_agent_health_norm if inspect_known and inspect_kind == "agent" else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "inspect_agent_has_melee",
        1.0 if inspect_known and inspect_kind == "agent" and agent.last_inspected_agent_has_melee else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "inspect_agent_has_ranged",
        1.0 if inspect_known and inspect_kind == "agent" and agent.last_inspected_agent_has_ranged else 0.0,
    )
    _append_feature(
        feature_names,
        feature_vector,
        "inspect_freshness_norm",
        _normalize_inspection_freshness(agent),
    )

    _append_feature(feature_names, feature_vector, "damaged_last_step", self_state.damaged_last_step)
    _append_feature(feature_names, feature_vector, "damage_from_up", self_state.damage_from_up)
    _append_feature(feature_names, feature_vector, "damage_from_down", self_state.damage_from_down)
    _append_feature(feature_names, feature_vector, "damage_from_left", self_state.damage_from_left)
    _append_feature(feature_names, feature_vector, "damage_from_right", self_state.damage_from_right)
    _append_feature(
        feature_names,
        feature_vector,
        "low_health_state",
        1.0 if self_state.health_norm <= config.LOW_HEALTH_THRESHOLD else 0.0,
    )

    # "can_attack" tells the future policy whether a melee action could succeed
    # right now without needing to infer adjacency from multiple separate features.
    can_attack = 1.0 if any(
        target is not None and target.entity_id != agent.entity_id
        for target in (
            world.get_agent_at(agent.x, agent.y - 1),
            world.get_agent_at(agent.x, agent.y + 1),
            world.get_agent_at(agent.x - 1, agent.y),
            world.get_agent_at(agent.x + 1, agent.y),
        )
    ) else 0.0
    _append_feature(feature_names, feature_vector, "can_attack", can_attack)
    _append_feature(feature_names, feature_vector, "visible_enemy_in_melee_range", can_attack)
    can_shoot = 1.0 if any(
        world._find_shot_target(agent, action) is not None
        for action in SHOOT_ACTIONS
    ) else 0.0
    _append_feature(feature_names, feature_vector, "can_shoot", can_shoot)
    _append_feature(feature_names, feature_vector, "visible_enemy_in_shoot_line", can_shoot)
    _append_feature(
        feature_names,
        feature_vector,
        "visible_agent_count_norm",
        len(visible_agents) / max(1, world.num_agents - 1),
    )
    _append_feature(
        feature_names,
        feature_vector,
        "visible_cell_ratio",
        len(visible_cells) / max(1, (2 * radius + 1) ** 2),
    )

    return feature_names, feature_vector


def build_self_state(world, agent) -> SelfState:
    """Build the self-only observation features for one agent."""

    damage_direction = _normalize_damage_direction(agent.last_damage_direction)
    return SelfState(
        x_norm=agent.x / max(1, world.width - 1),
        y_norm=agent.y / max(1, world.height - 1),
        health_norm=agent.health / max(1, agent.max_health),
        damaged_last_step=1.0 if agent.damaged_last_step else 0.0,
        damage_from_up=1.0 if damage_direction == "UP" else 0.0,
        damage_from_down=1.0 if damage_direction == "DOWN" else 0.0,
        damage_from_left=1.0 if damage_direction == "LEFT" else 0.0,
        damage_from_right=1.0 if damage_direction == "RIGHT" else 0.0,
    )


def build_observation(world, agent, radius: int) -> ObservationSnapshot:
    """Build the full observation package used for debugging and later learning."""

    visible_cells = _get_visible_cells(world, agent, radius)
    local_grid = build_local_grid(world, agent, radius, visible_cells=visible_cells)
    feature_names, feature_vector = build_feature_vector_from_visible(
        world,
        agent,
        radius,
        visible_cells,
    )
    visible_agents = _visible_other_agents(world, agent, visible_cells)
    nearest_agent = _nearest_visible_agent(agent, visible_agents)

    return ObservationSnapshot(
        agent_id=agent.entity_id,
        visible_cells=visible_cells,
        local_grid=local_grid,
        local_grid_lines=format_local_grid(local_grid),
        self_state=build_self_state(world, agent),
        feature_names=feature_names,
        feature_vector=feature_vector,
        nearest_visible_agent_id=None if nearest_agent is None else nearest_agent.entity_id,
        nearest_visible_agent_dx=None if nearest_agent is None else nearest_agent.x - agent.x,
        nearest_visible_agent_dy=None if nearest_agent is None else nearest_agent.y - agent.y,
        nearest_visible_agent_distance=(
            None
            if nearest_agent is None
            else manhattan_distance((agent.x, agent.y), (nearest_agent.x, nearest_agent.y))
        ),
    )


def format_local_grid(local_grid: list[list[str]]) -> list[str]:
    """Return the local observation grid as printable monospace strings."""

    return [" ".join(row) for row in local_grid]


def format_feature_lines(feature_names: list[str], feature_vector: list[float]) -> list[str]:
    """Return the feature vector in a readable name=value format."""

    return [
        f"{name}={value:.2f}"
        for name, value in zip(feature_names, feature_vector, strict=False)
    ]


def _classify_visible_cell(world, agent, world_x: int, world_y: int) -> str:
    """Map a visible world cell into a compact symbolic observation token."""

    if (world_x, world_y) == (agent.x, agent.y):
        return SELF_CELL
    if world.grid_map.is_wall(world_x, world_y):
        return WALL_CELL
    occupant = world.get_agent_at(world_x, world_y)
    if occupant is not None and occupant.entity_id != agent.entity_id:
        return AGENT_CELL
    item = world.get_item_at(world_x, world_y)
    if item is not None:
        if item.item_type == "heal":
            return HEAL_ITEM_CELL
        if item.item_type == "melee_weapon":
            return MELEE_WEAPON_CELL
        return RANGED_WEAPON_CELL
    return EMPTY_CELL


def _visible_other_agents(world, agent, visible_cells: set[tuple[int, int]]) -> list:
    """Return living non-self agents that are inside the visible set."""

    return [
        other
        for other in world.alive_agents()
        if other.entity_id != agent.entity_id and (other.x, other.y) in visible_cells
    ]


def _visible_items(world, visible_cells: set[tuple[int, int]]) -> list:
    """Return item entities that are visible to the current observing agent."""

    return [
        item
        for item in world.items.values()
        if item.alive and (item.x, item.y) in visible_cells
    ]


def _nearest_visible_agent(agent, visible_agents: list):
    """Return the closest visible other agent, or None if none are visible."""

    if not visible_agents:
        return None
    return min(
        visible_agents,
        key=lambda other: (
            manhattan_distance((agent.x, agent.y), (other.x, other.y)),
            other.entity_id,
        ),
    )


def _nearest_visible_item(agent, visible_items: list):
    """Return the closest visible item, or None when no item is visible."""

    if not visible_items:
        return None
    return min(
        visible_items,
        key=lambda item: (
            manhattan_distance((agent.x, agent.y), (item.x, item.y)),
            item.entity_id,
        ),
    )


def _normalize_relative_axis(delta: int, scale: int) -> float:
    """Normalize a signed relative axis value into the range [-1, 1]."""

    return delta / max(1, scale - 1)


def _get_visible_cells(world, agent, radius: int) -> set[tuple[int, int]]:
    """Use a world-level visibility cache when available, otherwise compute directly."""

    if hasattr(world, "get_visible_cells"):
        return world.get_visible_cells(agent, radius)
    return compute_visible_cells(world, agent, radius)


def _normalize_equipped_weapon_charges(agent) -> float:
    """Normalize remaining equipped weapon charges into the range [0, 1]."""

    if agent.equipped_weapon_type == "melee_weapon":
        return agent.equipped_weapon_charges / max(1, config.MELEE_WEAPON_ATTACK_CHARGES)
    if agent.equipped_weapon_type == "ranged_weapon":
        return agent.equipped_weapon_charges / max(1, config.RANGED_WEAPON_SHOT_CHARGES)
    return 0.0


def _normalize_inspection_freshness(agent) -> float:
    """Normalize how fresh the current short-lived inspection memory is."""

    return agent.last_inspection_age / max(1, config.INSPECT_MEMORY_TICKS)


def _normalize_damage_direction(last_damage_direction: str | None) -> str | None:
    """Convert attack action names into the direction damage came from."""

    mapping = {
        "ATTACK_UP": "DOWN",
        "ATTACK_DOWN": "UP",
        "ATTACK_LEFT": "RIGHT",
        "ATTACK_RIGHT": "LEFT",
        "SHOOT_UP": "DOWN",
        "SHOOT_DOWN": "UP",
        "SHOOT_LEFT": "RIGHT",
        "SHOOT_RIGHT": "LEFT",
    }
    return mapping.get(last_damage_direction)


def _append_feature(
    feature_names: list[str],
    feature_vector: list[float],
    name: str,
    value: float,
) -> None:
    """Append one named feature to the observation vector."""

    feature_names.append(name)
    feature_vector.append(value)
