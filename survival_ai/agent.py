"""Controller implementations for scripted and random Phase 1 agents."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from . import config
from .actions import Action, MOVEMENT_ACTIONS, SHOOT_ACTIONS, action_to_delta
from .network import masked_softmax, select_index_from_probabilities
from .observation import build_observation, compute_visible_cells
from .utils import manhattan_distance


class AgentController(ABC):
    """Abstract action-selection interface for a world agent."""

    @abstractmethod
    def choose_action(self, world, agent) -> Action:
        """Choose the next action for the given agent."""


class RandomController(AgentController):
    """Chooses uniformly from legal actions in the current state."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, world, agent) -> Action:
        legal_actions = world.get_legal_actions(agent)
        return self._rng.choice(legal_actions)


class ScriptedController(AgentController):
    """Uses simple heuristics to seek, move, and attack visible opponents."""

    def __init__(self, vision_radius: int, seed: int | None = None) -> None:
        self._vision_radius = vision_radius
        self._rng = random.Random(seed)
        self._previous_position: tuple[int, int] | None = None

    def choose_action(self, world, agent) -> Action:
        current_position = (agent.x, agent.y)
        legal_actions = world.get_legal_actions(agent)
        visible_cells = compute_visible_cells(world, agent, self._vision_radius)
        adjacent_attack = self._find_adjacent_attack(world, agent, legal_actions)
        if adjacent_attack is not None:
            self._previous_position = current_position
            return adjacent_attack

        ranged_attack = self._find_ranged_attack(world, agent, legal_actions)
        if ranged_attack is not None:
            self._previous_position = current_position
            return ranged_attack

        if (
            Action.USE_ITEM in legal_actions
            and agent.inventory_item_type == "heal"
            and agent.health <= max(1, agent.max_health // 2)
        ):
            self._previous_position = current_position
            return Action.USE_ITEM

        if (
            Action.USE_ITEM in legal_actions
            and agent.inventory_item_type in {"melee_weapon", "ranged_weapon"}
            and not agent.has_equipped_weapon()
        ):
            self._previous_position = current_position
            return Action.USE_ITEM

        if self._should_drop_for_weapon_swap(world, agent, legal_actions, visible_cells):
            self._previous_position = current_position
            return Action.DROP_ITEM

        if Action.PICKUP in legal_actions and not agent.has_inventory_item():
            self._previous_position = current_position
            return Action.PICKUP

        visible_targets = [
            other
            for other in world.alive_agents()
            if other.entity_id != agent.entity_id and (other.x, other.y) in visible_cells
        ]

        if visible_targets:
            nearest_target = min(
                visible_targets,
                key=lambda other: (
                    manhattan_distance((agent.x, agent.y), (other.x, other.y)),
                    other.entity_id,
                ),
            )
            movement_action = self._choose_movement_toward(agent, nearest_target, legal_actions)
            if movement_action is not None:
                self._previous_position = current_position
                return movement_action

        legal_movement = [action for action in legal_actions if action in MOVEMENT_ACTIONS]
        if legal_movement:
            action = self._rng.choice(legal_movement)
            self._previous_position = current_position
            return action
        self._previous_position = current_position
        return Action.WAIT

    def _find_adjacent_attack(self, world, agent, legal_actions: list[Action]) -> Action | None:
        """Return an attack action when a target is adjacent to the agent."""

        for action in legal_actions:
            if action not in (
                Action.ATTACK_UP,
                Action.ATTACK_DOWN,
                Action.ATTACK_LEFT,
                Action.ATTACK_RIGHT,
            ):
                continue
            dx, dy = action_to_delta(action)
            target = world.get_agent_at(agent.x + dx, agent.y + dy)
            if target is not None and target.entity_id != agent.entity_id:
                return action
        return None

    def _find_ranged_attack(self, world, agent, legal_actions: list[Action]) -> Action | None:
        """Return a shoot action when a target is lined up for the ranged weapon."""

        if not agent.has_ranged_weapon():
            return None
        for action in legal_actions:
            if action not in SHOOT_ACTIONS:
                continue
            if world._find_shot_target(agent, action) is not None:
                return action
        return None

    @staticmethod
    def _should_drop_for_weapon_swap(
        world,
        agent,
        legal_actions: list[Action],
        visible_cells: set[tuple[int, int]],
    ) -> bool:
        """Return True when dropping the current weapon would enable a useful swap."""

        if (
            Action.DROP_ITEM not in legal_actions
            or agent.has_inventory_item()
            or not agent.has_equipped_weapon()
        ):
            return False

        desired_item_type = None
        if agent.has_ranged_weapon():
            desired_item_type = "melee_weapon"
        elif agent.has_active_weapon_bonus():
            desired_item_type = "ranged_weapon"
        else:
            return False

        return any(
            item.alive
            and item.item_type == desired_item_type
            and (item.x, item.y) in visible_cells
            for item in world.items.values()
        )

    def _choose_movement_toward(self, agent, target, legal_actions: list[Action]) -> Action | None:
        """Select a legal move that reduces Manhattan distance to the target."""

        scored_moves: list[tuple[tuple[int, int, int, float], Action]] = []
        current_distance = manhattan_distance((agent.x, agent.y), (target.x, target.y))

        for action in legal_actions:
            if action not in MOVEMENT_ACTIONS:
                continue

            move_dx, move_dy = action_to_delta(action)
            next_position = (agent.x + move_dx, agent.y + move_dy)
            next_distance = manhattan_distance(next_position, (target.x, target.y))
            if next_distance > current_distance:
                continue

            diagonal_target = abs(target.x - agent.x) == 1 and abs(target.y - agent.y) == 1
            axis_preference = self._axis_preference(agent, target)
            is_preferred_axis = 0
            if diagonal_target:
                if axis_preference == "horizontal" and move_dx != 0:
                    is_preferred_axis = 1
                elif axis_preference == "vertical" and move_dy != 0:
                    is_preferred_axis = 1

            backtrack_penalty = 1 if next_position == self._previous_position else 0
            random_tiebreak = self._rng.random()
            score = (
                next_distance,
                -is_preferred_axis,
                backtrack_penalty,
                random_tiebreak,
            )
            scored_moves.append((score, action))

        if not scored_moves:
            return None

        scored_moves.sort(key=lambda item: item[0])
        return scored_moves[0][1]

    @staticmethod
    def _axis_preference(agent, target) -> str:
        """Choose a deterministic axis preference to break mirrored movement ties."""

        return "horizontal" if agent.entity_id < target.entity_id else "vertical"


class PolicyController(AgentController):
    """Chooses actions from the learned policy network using current observations."""

    ACTION_SELECTION_MODES = (
        "greedy",
        "sample",
        "temperature_sample",
        "epsilon_greedy",
        "top_k_sample",
        "temperature_top_k_sample",
    )

    def __init__(
        self,
        network,
        vision_radius: int,
        mode: str = "greedy",
        seed: int | None = None,
    ) -> None:
        self._network = network
        self._vision_radius = vision_radius
        if mode not in self.ACTION_SELECTION_MODES:
            raise ValueError(
                f"Unsupported policy mode '{mode}'. Expected one of {self.ACTION_SELECTION_MODES}."
            )
        self._mode = mode
        self._rng = random.Random(seed)

    def choose_action(self, world, agent) -> Action:
        observation = build_observation(world, agent, self._vision_radius)
        legal_actions = world.get_legal_actions(agent)
        legal_indices = [action.value for action in legal_actions]
        action_scores = self._network.forward(observation.feature_vector)
        probabilities = masked_softmax(action_scores, legal_indices)

        if self._mode == "sample":
            chosen_index = select_index_from_probabilities(probabilities, self._rng)
        elif self._mode == "temperature_sample":
            chosen_index = self._choose_temperature_sample(action_scores, legal_indices)
        elif self._mode == "epsilon_greedy":
            chosen_index = self._choose_epsilon_greedy(probabilities, legal_indices)
        elif self._mode == "top_k_sample":
            chosen_index = self._choose_top_k_sample(action_scores, legal_indices)
        elif self._mode == "temperature_top_k_sample":
            chosen_index = self._choose_temperature_top_k_sample(action_scores, legal_indices)
        else:
            chosen_index = max(legal_indices, key=lambda index: probabilities[index])
        return Action(chosen_index)

    @classmethod
    def available_modes(cls) -> tuple[str, ...]:
        """Return the supported policy playback modes."""

        return cls.ACTION_SELECTION_MODES

    def _choose_temperature_sample(
        self,
        action_scores: list[float],
        legal_indices: list[int],
    ) -> int:
        """Sample from a softened or sharpened action distribution."""

        temperature = max(1e-6, config.POLICY_TEMPERATURE)
        scaled_scores = list(action_scores)
        for index in legal_indices:
            scaled_scores[index] = action_scores[index] / temperature
        probabilities = masked_softmax(scaled_scores, legal_indices)
        return select_index_from_probabilities(probabilities, self._rng)

    def _choose_epsilon_greedy(
        self,
        probabilities: list[float],
        legal_indices: list[int],
    ) -> int:
        """Usually pick the best legal action, but occasionally explore randomly."""

        if self._rng.random() < config.POLICY_EPSILON:
            return self._rng.choice(legal_indices)
        return max(legal_indices, key=lambda index: probabilities[index])

    def _choose_top_k_sample(
        self,
        action_scores: list[float],
        legal_indices: list[int],
    ) -> int:
        """Sample only among the highest-scoring legal actions."""

        top_k = max(1, min(config.POLICY_TOP_K, len(legal_indices)))
        ranked_indices = sorted(
            legal_indices,
            key=lambda index: action_scores[index],
            reverse=True,
        )
        top_indices = ranked_indices[:top_k]
        probabilities = masked_softmax(action_scores, top_indices)
        return select_index_from_probabilities(probabilities, self._rng)

    def _choose_temperature_top_k_sample(
        self,
        action_scores: list[float],
        legal_indices: list[int],
    ) -> int:
        """Apply temperature scaling first, then sample within the top-k legal actions."""

        temperature = max(1e-6, config.POLICY_TEMPERATURE)
        scaled_scores = {
            index: action_scores[index] / temperature
            for index in legal_indices
        }
        top_k = max(1, min(config.POLICY_TOP_K, len(legal_indices)))
        ranked_indices = sorted(
            legal_indices,
            key=lambda index: scaled_scores[index],
            reverse=True,
        )
        top_indices = ranked_indices[:top_k]
        scaled_logits = [0.0 for _ in action_scores]
        for index in top_indices:
            scaled_logits[index] = scaled_scores[index]
        probabilities = masked_softmax(scaled_logits, top_indices)
        return select_index_from_probabilities(probabilities, self._rng)
