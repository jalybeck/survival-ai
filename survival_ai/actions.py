"""Discrete action definitions and helpers for the survival world."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Enumerates the available actions for Phase 1 agents."""

    WAIT = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ATTACK_UP = 5
    ATTACK_DOWN = 6
    ATTACK_LEFT = 7
    ATTACK_RIGHT = 8
    PICKUP = 9
    USE_ITEM = 10
    DROP_ITEM = 11
    SHOOT_UP = 12
    SHOOT_DOWN = 13
    SHOOT_LEFT = 14
    SHOOT_RIGHT = 15


MOVEMENT_ACTIONS = (
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
)

ATTACK_ACTIONS = (
    Action.ATTACK_UP,
    Action.ATTACK_DOWN,
    Action.ATTACK_LEFT,
    Action.ATTACK_RIGHT,
)

ITEM_ACTIONS = (
    Action.PICKUP,
    Action.USE_ITEM,
    Action.DROP_ITEM,
)

SHOOT_ACTIONS = (
    Action.SHOOT_UP,
    Action.SHOOT_DOWN,
    Action.SHOOT_LEFT,
    Action.SHOOT_RIGHT,
)

ACTION_TO_DELTA = {
    Action.MOVE_UP: (0, -1),
    Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0),
    Action.MOVE_RIGHT: (1, 0),
    Action.ATTACK_UP: (0, -1),
    Action.ATTACK_DOWN: (0, 1),
    Action.ATTACK_LEFT: (-1, 0),
    Action.ATTACK_RIGHT: (1, 0),
    Action.SHOOT_UP: (0, -1),
    Action.SHOOT_DOWN: (0, 1),
    Action.SHOOT_LEFT: (-1, 0),
    Action.SHOOT_RIGHT: (1, 0),
}


def is_movement_action(action: Action) -> bool:
    """Return True when the action moves the agent by one tile."""

    return action in MOVEMENT_ACTIONS


def is_attack_action(action: Action) -> bool:
    """Return True when the action attacks an adjacent tile."""

    return action in ATTACK_ACTIONS


def is_item_action(action: Action) -> bool:
    """Return True when the action operates on the agent inventory or current tile."""

    return action in ITEM_ACTIONS


def is_shoot_action(action: Action) -> bool:
    """Return True when the action fires a ranged weapon in one direction."""

    return action in SHOOT_ACTIONS


def action_to_delta(action: Action) -> tuple[int, int]:
    """Translate a directional action into a coordinate delta."""

    return ACTION_TO_DELTA.get(action, (0, 0))
