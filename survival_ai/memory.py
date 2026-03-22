"""Episode memory structures for storing trajectories during policy learning."""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import Action


@dataclass(slots=True)
class EpisodeStep:
    """Stores one policy-learning transition for a single agent."""

    agent_id: int
    observation: list[float]
    legal_action_indices: list[int]
    action: Action
    reward: float
    done: bool


@dataclass(slots=True)
class EpisodeMemory:
    """Collects trajectory steps per agent for one completed episode."""

    steps_by_agent: dict[int, list[EpisodeStep]] = field(default_factory=dict)

    def add_step(self, step: EpisodeStep) -> None:
        """Append a transition to the selected agent's trajectory."""

        self.steps_by_agent.setdefault(step.agent_id, []).append(step)

    def get_agent_steps(self, agent_id: int) -> list[EpisodeStep]:
        """Return all stored steps for the given agent id."""

        return self.steps_by_agent.get(agent_id, [])

    def all_steps(self) -> list[EpisodeStep]:
        """Flatten all stored trajectories into one list."""

        flattened: list[EpisodeStep] = []
        for steps in self.steps_by_agent.values():
            flattened.extend(steps)
        return flattened

    def clear(self) -> None:
        """Remove all stored episode data."""

        self.steps_by_agent.clear()
