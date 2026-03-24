"""Episode-based training loop that connects observation, action, reward, and updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from . import config
from .actions import Action
from .memory import EpisodeMemory, EpisodeStep
from .network import SimpleMLP, masked_softmax, select_index_from_probabilities
from .observation import build_feature_vector
from .reward import RewardBreakdown, compute_step_rewards
from .world import World


@dataclass(slots=True)
class EpisodeMetrics:
    """Summarizes the outcome of one completed training episode."""

    episode_index: int
    ticks: int
    winner_id: int | None
    reward_totals: dict[int, float]
    mean_reward: float
    reward_component_totals: RewardBreakdown


@dataclass(slots=True)
class DecisionContext:
    """Stores one agent decision input so training code can reuse it cleanly."""

    observation: list[float]
    legal_action_indices: list[int]


class Trainer:
    """Coordinates self-play data collection and policy gradient updates."""

    def __init__(
        self,
        world: World,
        policy_network: SimpleMLP,
        vision_radius: int = config.VISION_RADIUS,
        discount_factor: float = config.DISCOUNT_FACTOR,
        learning_rate: float = config.LEARNING_RATE,
        weights_path: str = config.WEIGHTS_PATH,
        seed: int | None = None,
        randomize_seed_each_episode: bool = False,
    ) -> None:
        self.world = world
        self.policy_network = policy_network
        self.vision_radius = vision_radius
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.weights_path = Path(weights_path)
        self._rng = random.Random(seed)
        self._system_rng = random.SystemRandom()
        self.randomize_seed_each_episode = randomize_seed_each_episode

    def run_episode(self, episode_index: int = 0) -> tuple[EpisodeMemory, EpisodeMetrics]:
        """Collect one complete self-play episode using the current policy."""

        if self.randomize_seed_each_episode:
            self._rng.seed(self._system_rng.randint(0, 2**31 - 1))
        self.world.reset()
        memory = EpisodeMemory()
        component_totals = RewardBreakdown()

        while True:
            alive_ids = [agent.entity_id for agent in self.world.alive_agents()]
            decision_contexts: dict[int, DecisionContext] = {}
            selected_actions: dict[int, Action] = {}

            for agent_id in alive_ids:
                agent = self.world.agents[agent_id]
                decision_context = self._build_decision_context(agent)
                decision_contexts[agent_id] = decision_context
                selected_actions[agent_id] = self._sample_action(
                    decision_context.observation,
                    decision_context.legal_action_indices,
                )

            step_result = self.world.step(selected_actions)
            reward_breakdowns = compute_step_rewards(self.world, step_result)
            self._accumulate_reward_components(component_totals, reward_breakdowns)

            for agent_id in alive_ids:
                reward_value = reward_breakdowns[agent_id].total
                decision_context = decision_contexts[agent_id]
                memory.add_step(
                    EpisodeStep(
                        agent_id=agent_id,
                        observation=decision_context.observation,
                        legal_action_indices=decision_context.legal_action_indices,
                        action=selected_actions[agent_id],
                        reward=reward_value,
                        done=(not self.world.agents[agent_id].alive) or step_result.episode_over,
                    )
                )

                agent = self.world.agents[agent_id]
                agent.last_reward = reward_value
                agent.total_reward += reward_value

            if step_result.episode_over:
                break

        metrics = EpisodeMetrics(
            episode_index=episode_index,
            ticks=self.world.tick,
            winner_id=step_result.winner_id,
            reward_totals={
                agent_id: self.world.agents[agent_id].total_reward
                for agent_id in self.world.agents
            },
            mean_reward=sum(agent.total_reward for agent in self.world.agents.values()) / max(1, len(self.world.agents)),
            reward_component_totals=component_totals,
        )
        return memory, metrics

    def train(self, num_episodes: int, log_every: int = 10, save_every: int = 50) -> None:
        """Run repeated self-play episodes and update the shared policy network."""

        for episode_index in range(1, num_episodes + 1):
            episode_memory, metrics = self.run_episode(episode_index=episode_index)
            self.update_policy(episode_memory)

            if episode_index % log_every == 0 or episode_index == 1:
                print(
                    f"Episode {episode_index:>4} | "
                    f"ticks={metrics.ticks:>3} | "
                    f"winner={metrics.winner_id} | "
                    f"mean_reward={metrics.mean_reward:+.3f} | "
                    f"rewards={{{', '.join(f'{agent_id}:{value:+.2f}' for agent_id, value in metrics.reward_totals.items())}}} | "
                    f"components={self._format_component_totals(metrics.reward_component_totals)}"
                )

            if episode_index % save_every == 0:
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                self.policy_network.save(str(self.weights_path))

        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_network.save(str(self.weights_path))

    def compute_returns(self, episode_memory: EpisodeMemory) -> dict[int, list[float]]:
        """Compute discounted returns for each agent trajectory."""

        returns_by_agent: dict[int, list[float]] = {}

        for agent_id, steps in episode_memory.steps_by_agent.items():
            running_return = 0.0
            reversed_returns: list[float] = []
            for step in reversed(steps):
                running_return = step.reward + self.discount_factor * running_return
                reversed_returns.append(running_return)
            returns_by_agent[agent_id] = list(reversed(reversed_returns))

        return returns_by_agent

    def update_policy(self, episode_memory: EpisodeMemory) -> None:
        """Apply one REINFORCE-style policy update using stored episode returns."""

        returns_by_agent = self.compute_returns(episode_memory)
        all_returns = [value for returns in returns_by_agent.values() for value in returns]
        if not all_returns:
            return

        normalized_returns = self._normalize_returns(all_returns)
        normalized_iter = iter(normalized_returns)
        total_steps = len(all_returns)

        for agent_id, steps in episode_memory.steps_by_agent.items():
            for step in steps:
                scaled_return = next(normalized_iter)
                action_scores = self.policy_network.forward(step.observation)
                probabilities = masked_softmax(action_scores, step.legal_action_indices)

                # This is the policy-gradient core: increase probability for
                # actions with positive return and decrease it for actions with
                # negative return. Scaling by the number of steps keeps updates
                # from exploding when an episode is long.
                output_gradients = list(probabilities)
                output_gradients[step.action.value] -= 1.0
                scale = scaled_return / max(1, total_steps)
                output_gradients = [gradient * scale for gradient in output_gradients]
                self.policy_network.backward(output_gradients)

        self.policy_network.update(self.learning_rate)

    def load_weights_if_available(self) -> bool:
        """Load saved network weights if the configured file exists and looks usable."""

        if not self.weights_path.exists():
            return False
        try:
            self.policy_network.load(str(self.weights_path))
        except (ValueError, KeyError, TypeError):
            return False
        return True

    def _sample_action(self, observation_vector: list[float], legal_action_indices: list[int]) -> Action:
        """Sample one legal action from the current policy distribution."""

        action_scores = self.policy_network.forward(observation_vector)
        probabilities = masked_softmax(action_scores, legal_action_indices)
        chosen_index = select_index_from_probabilities(probabilities, self._rng)
        return Action(chosen_index)

    def _build_decision_context(self, agent) -> DecisionContext:
        """Build one reusable decision package for the current agent state."""

        _feature_names, feature_vector = build_feature_vector(
            self.world,
            agent,
            self.vision_radius,
        )
        legal_actions = self.world.get_legal_actions(agent)
        return DecisionContext(
            observation=feature_vector,
            legal_action_indices=[action.value for action in legal_actions],
        )

    @staticmethod
    def _normalize_returns(returns: list[float]) -> list[float]:
        """Normalize returns to stabilize the early learning signal."""

        if not returns:
            return []
        mean_value = sum(returns) / len(returns)
        variance = sum((value - mean_value) ** 2 for value in returns) / len(returns)
        std_value = variance ** 0.5
        if std_value < 1e-8:
            return [value - mean_value for value in returns]
        return [(value - mean_value) / std_value for value in returns]

    @staticmethod
    def _accumulate_reward_components(
        totals: RewardBreakdown,
        reward_breakdowns: dict[int, RewardBreakdown],
    ) -> None:
        """Sum all reward components over the full episode for logging."""

        for breakdown in reward_breakdowns.values():
            totals.total += breakdown.total
            totals.survive_reward += breakdown.survive_reward
            totals.damage_reward += breakdown.damage_reward
            totals.damage_penalty += breakdown.damage_penalty
            totals.death_penalty += breakdown.death_penalty
            totals.win_reward += breakdown.win_reward
            totals.exploration_reward += breakdown.exploration_reward
            totals.approach_reward += breakdown.approach_reward
            totals.attack_range_reward += breakdown.attack_range_reward
            totals.item_pickup_reward += breakdown.item_pickup_reward
            totals.contextual_weapon_pickup_reward += breakdown.contextual_weapon_pickup_reward
            totals.contextual_heal_pickup_reward += breakdown.contextual_heal_pickup_reward
            totals.heal_item_use_reward += breakdown.heal_item_use_reward
            totals.low_health_heal_bonus += breakdown.low_health_heal_bonus
            totals.low_health_retreat_reward += breakdown.low_health_retreat_reward
            totals.low_health_approach_penalty += breakdown.low_health_approach_penalty
            totals.low_health_melee_range_penalty += breakdown.low_health_melee_range_penalty
            totals.weapon_item_use_reward += breakdown.weapon_item_use_reward
            totals.weapon_hit_bonus += breakdown.weapon_hit_bonus
            totals.weapon_kill_bonus += breakdown.weapon_kill_bonus
            totals.armed_visible_enemy_reward += breakdown.armed_visible_enemy_reward
            totals.drop_penalty += breakdown.drop_penalty
            totals.drop_weapon_while_threatened_penalty += breakdown.drop_weapon_while_threatened_penalty
            totals.idle_penalty += breakdown.idle_penalty
            totals.oscillation_penalty += breakdown.oscillation_penalty
            totals.no_winner_penalty += breakdown.no_winner_penalty

    @staticmethod
    def _format_component_totals(component_totals: RewardBreakdown) -> str:
        """Format the episode-wide reward component totals for compact logging."""

        return (
            f"survive={component_totals.survive_reward:+.2f}, "
            f"deal={component_totals.damage_reward:+.2f}, "
            f"take={component_totals.damage_penalty:+.2f}, "
            f"death={component_totals.death_penalty:+.2f}, "
            f"win={component_totals.win_reward:+.2f}, "
            f"explore={component_totals.exploration_reward:+.2f}, "
            f"approach={component_totals.approach_reward:+.2f}, "
            f"range={component_totals.attack_range_reward:+.2f}, "
            f"pickup={component_totals.item_pickup_reward:+.2f}, "
            f"pickup_weapon_ctx={component_totals.contextual_weapon_pickup_reward:+.2f}, "
            f"pickup_heal_ctx={component_totals.contextual_heal_pickup_reward:+.2f}, "
            f"heal={component_totals.heal_item_use_reward:+.2f}, "
            f"heal_low={component_totals.low_health_heal_bonus:+.2f}, "
            f"retreat_low={component_totals.low_health_retreat_reward:+.2f}, "
            f"approach_low={component_totals.low_health_approach_penalty:+.2f}, "
            f"melee_low={component_totals.low_health_melee_range_penalty:+.2f}, "
            f"weapon={component_totals.weapon_item_use_reward:+.2f}, "
            f"weapon_hit={component_totals.weapon_hit_bonus:+.2f}, "
            f"weapon_kill={component_totals.weapon_kill_bonus:+.2f}, "
            f"armed={component_totals.armed_visible_enemy_reward:+.2f}, "
            f"drop={component_totals.drop_penalty:+.2f}, "
            f"drop_threat={component_totals.drop_weapon_while_threatened_penalty:+.2f}, "
            f"idle={component_totals.idle_penalty:+.2f}, "
            f"loop={component_totals.oscillation_penalty:+.2f}, "
            f"draw={component_totals.no_winner_penalty:+.2f}"
        )
