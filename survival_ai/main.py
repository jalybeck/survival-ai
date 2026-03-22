"""Application entrypoint for scripted debug, policy playback, and training."""

from __future__ import annotations

import random
import sys
from typing import Callable

from . import config
from .actions import Action
from .agent import PolicyController, ScriptedController
from .memory import EpisodeMemory, EpisodeStep
from .network import SimpleMLP
from .observation import build_observation, format_feature_lines
from .reward import (
    RewardBreakdown,
    compute_step_rewards,
    create_empty_reward_breakdowns,
    format_reward_breakdown,
)
from .trainer import Trainer
from .world import World


def build_scripted_controllers(world: World, seed: int) -> dict[int, ScriptedController]:
    """Create deterministic scripted controllers for all agents."""

    return {
        agent_id: ScriptedController(
            vision_radius=config.VISION_RADIUS,
            seed=seed + agent_id,
        )
        for agent_id in world.agents
    }


def build_policy_controllers(
    world: World,
    network: SimpleMLP,
    seed: int,
    mode: str,
) -> dict[int, PolicyController]:
    """Create greedy policy controllers that use the learned network."""

    return {
        agent_id: PolicyController(
            network=network,
            vision_radius=config.VISION_RADIUS,
            mode=mode,
            seed=seed + agent_id,
        )
        for agent_id in world.agents
    }


def print_observation_snapshot(observation_snapshot, reward_breakdown=None) -> None:
    """Print the selected agent observation in a readable terminal format."""

    print()
    print(f"Observation for agent {observation_snapshot.agent_id}")
    if reward_breakdown is not None:
        print(f"Reward: {format_reward_breakdown(reward_breakdown)}")
    print("Local grid:")
    for row in observation_snapshot.local_grid_lines:
        print(f"  {row}")
    print("Features:")
    for line in format_feature_lines(
        observation_snapshot.feature_names,
        observation_snapshot.feature_vector,
    ):
        print(f"  {line}")


def format_action_scores(action_scores: list[float]) -> list[str]:
    """Format raw network action scores for debug output."""

    ranked_actions = sorted(
        ((Action(index), score) for index, score in enumerate(action_scores)),
        key=lambda item: item[1],
        reverse=True,
    )
    return [f"{action.name}={score:.3f}" for action, score in ranked_actions]


def create_network_for_world(world: World, seed: int = config.DEFAULT_SEED) -> SimpleMLP:
    """Create a network whose input size matches the current observation vector."""

    sample_agent = world.agents[min(world.agents)]
    initial_observation = build_observation(world, sample_agent, config.VISION_RADIUS)
    return SimpleMLP(
        input_size=len(initial_observation.feature_vector),
        hidden_sizes=config.HIDDEN_LAYER_SIZES,
        output_size=len(Action),
        seed=seed,
    )


def build_network_debug_snapshot(
    world: World,
    debug_network: SimpleMLP,
    agent_id: int,
) -> tuple[object, object]:
    """Build the selected agent observation and forward trace for debug rendering."""

    observation_snapshot = build_observation(
        world,
        world.agents[agent_id],
        config.VISION_RADIUS,
    )
    network_trace = debug_network.inspect_forward(observation_snapshot.feature_vector)
    return observation_snapshot, network_trace


def run_debug_loop(
    world: World,
    controllers,
    debug_network: SimpleMLP,
    controller_builder,
    controller_mode_label_getter: Callable[[], str | None] | None = None,
    controller_mode_cycle_callback: Callable[[], None] | None = None,
    episode_reset_callback: Callable[[], None] | None = None,
    live_trainer: Trainer | None = None,
) -> None:
    """Run the pygame loop for either scripted or learned policy playback."""

    from .render import PygameRenderer, pygame

    if pygame is None:
        raise RuntimeError("pygame is not installed. Install it with `pip install pygame`.")

    renderer = PygameRenderer(world.width, world.height)
    debug_agent_id = min(world.agents)
    reset_at: int | None = None
    auto_advance = False
    step_requested = False
    latest_rewards = create_empty_reward_breakdowns(world.agents.keys())
    active_tracers: list[dict[str, object]] = []
    live_episode_memory = EpisodeMemory()
    live_component_totals = RewardBreakdown()
    live_episode_index = 1
    cached_debug_snapshots = {
        agent_id: build_network_debug_snapshot(world, debug_network, agent_id)
        for agent_id in world.agents
    }
    previous_debug_snapshots = {
        agent_id: None
        for agent_id in world.agents
    }
    last_snapshot_tick = world.tick
    running = True

    try:
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        step_requested = True
                    elif event.key == pygame.K_a:
                        auto_advance = not auto_advance
                    elif (
                        event.key == pygame.K_m
                        and controller_mode_cycle_callback is not None
                    ):
                        controller_mode_cycle_callback()
                        controllers.clear()
                        controllers.update(controller_builder(world))
                    elif event.key == pygame.K_r:
                        world.reset()
                        if episode_reset_callback is not None:
                            episode_reset_callback()
                        controllers.clear()
                        controllers.update(controller_builder(world))
                        latest_rewards = create_empty_reward_breakdowns(world.agents.keys())
                        active_tracers = []
                        live_episode_memory.clear()
                        live_component_totals = RewardBreakdown()
                        cached_debug_snapshots = {
                            agent_id: build_network_debug_snapshot(world, debug_network, agent_id)
                            for agent_id in world.agents
                        }
                        previous_debug_snapshots = {
                            agent_id: None
                            for agent_id in world.agents
                        }
                        last_snapshot_tick = world.tick
                        reset_at = None
                        step_requested = False
                    elif event.key == pygame.K_TAB:
                        agent_ids = sorted(world.agents)
                        current_index = agent_ids.index(debug_agent_id)
                        debug_agent_id = agent_ids[(current_index + 1) % len(agent_ids)]
                    elif event.key == pygame.K_o:
                        debug_observation, network_trace = cached_debug_snapshots[debug_agent_id]
                        action_scores = network_trace.outputs
                        print_observation_snapshot(
                            debug_observation,
                            latest_rewards.get(debug_agent_id),
                        )
                        print("Network scores:")
                        for line in format_action_scores(action_scores):
                            print(f"  {line}")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if renderer.handle_mouse_click(event.pos):
                        continue

            should_advance = auto_advance or step_requested
            active_tracers = [
                tracer for tracer in active_tracers
                if int(tracer["expires_at"]) > now
            ]

            if reset_at is None and should_advance:
                training_step_context: dict[int, tuple[list[float], list[int], Action]] = {}
                actions = {
                    agent_id: controller.choose_action(world, world.agents[agent_id])
                    for agent_id, controller in controllers.items()
                    if world.agents[agent_id].alive
                }
                if live_trainer is not None:
                    for agent_id, action in actions.items():
                        agent = world.agents[agent_id]
                        observation_snapshot = build_observation(
                            world,
                            agent,
                            config.VISION_RADIUS,
                        )
                        legal_actions = world.get_legal_actions(agent)
                        training_step_context[agent_id] = (
                            observation_snapshot.feature_vector,
                            [legal_action.value for legal_action in legal_actions],
                            action,
                        )
                result = world.step(actions)
                latest_rewards = compute_step_rewards(world, result)
                for damage_event in result.damage_events:
                    if not damage_event.direction.startswith("SHOOT_"):
                        continue
                    attacker = world.agents[damage_event.attacker_id]
                    active_tracers.append(
                        {
                            "start": (attacker.x, attacker.y),
                            "end": damage_event.target_position,
                            "color": config.RANGED_TRACER_COLOR,
                            "expires_at": now + config.TRACER_DURATION_MS,
                        }
                    )
                if live_trainer is not None:
                    live_trainer._accumulate_reward_components(
                        live_component_totals,
                        latest_rewards,
                    )
                for agent_id, breakdown in latest_rewards.items():
                    world.agents[agent_id].last_reward = breakdown.total
                    world.agents[agent_id].total_reward += breakdown.total
                if live_trainer is not None:
                    for agent_id, (
                        observation_vector,
                        legal_action_indices,
                        action,
                    ) in training_step_context.items():
                        live_episode_memory.add_step(
                            EpisodeStep(
                                agent_id=agent_id,
                                observation=observation_vector,
                                legal_action_indices=legal_action_indices,
                                action=action,
                                reward=latest_rewards[agent_id].total,
                                done=(not world.agents[agent_id].alive) or result.episode_over,
                            )
                        )
                step_requested = False
                if result.episode_over:
                    if live_trainer is not None:
                        live_trainer.update_policy(live_episode_memory)
                        live_trainer.weights_path.parent.mkdir(parents=True, exist_ok=True)
                        live_trainer.policy_network.save(str(live_trainer.weights_path))
                        mean_reward = (
                            sum(agent.total_reward for agent in world.agents.values())
                            / max(1, len(world.agents))
                        )
                        reward_totals = ", ".join(
                            f"{agent_id}:{agent.total_reward:+.2f}"
                            for agent_id, agent in world.agents.items()
                        )
                        print(
                            f"Live Episode {live_episode_index:>4} | "
                            f"ticks={result.tick:>3} | "
                            f"winner={result.winner_id} | "
                            f"mean_reward={mean_reward:+.3f} | "
                            f"rewards={{{reward_totals}}} | "
                            f"components={Trainer._format_component_totals(live_component_totals)}"
                        )
                        live_episode_index += 1
                        live_episode_memory = EpisodeMemory()
                        live_component_totals = RewardBreakdown()
                    reset_at = now + config.RESET_DELAY_MS
            elif reset_at is not None and now >= reset_at:
                world.reset()
                if episode_reset_callback is not None:
                    episode_reset_callback()
                controllers.clear()
                controllers.update(controller_builder(world))
                latest_rewards = create_empty_reward_breakdowns(world.agents.keys())
                active_tracers = []
                cached_debug_snapshots = {
                    agent_id: build_network_debug_snapshot(world, debug_network, agent_id)
                    for agent_id in world.agents
                }
                previous_debug_snapshots = {
                    agent_id: None
                    for agent_id in world.agents
                }
                last_snapshot_tick = world.tick
                reset_at = None
                step_requested = False

            if world.tick != last_snapshot_tick:
                for agent_id, agent in world.agents.items():
                    if agent.alive:
                        previous_debug_snapshots[agent_id] = cached_debug_snapshots[agent_id]
                        cached_debug_snapshots[agent_id] = build_network_debug_snapshot(
                            world,
                            debug_network,
                            agent_id,
                        )
                last_snapshot_tick = world.tick

            debug_observation, network_trace = cached_debug_snapshots[debug_agent_id]
            previous_debug_snapshot = previous_debug_snapshots[debug_agent_id]
            previous_observation = None
            previous_network_trace = None
            if previous_debug_snapshot is not None:
                previous_observation, previous_network_trace = previous_debug_snapshot
            action_scores = network_trace.outputs
            renderer.draw(
                world,
                visible_cells=debug_observation.visible_cells,
                observation_snapshot=debug_observation,
                previous_observation_snapshot=previous_observation,
                debug_agent_id=debug_agent_id,
                auto_advance=auto_advance,
                controller_mode_label=(
                    controller_mode_label_getter()
                    if controller_mode_label_getter is not None
                    else None
                ),
                debug_agent_dead=(not world.agents[debug_agent_id].alive),
                network_trace=network_trace,
                previous_network_trace=previous_network_trace,
                action_labels=[Action(index).name for index in range(len(action_scores))],
                network_debug_lines=format_action_scores(action_scores)[:9],
                tracer_lines=[
                    (
                        tuple(tracer["start"]),
                        tuple(tracer["end"]),
                        tuple(tracer["color"]),
                    )
                    for tracer in active_tracers
                ],
            )
            renderer.clock.tick(config.FPS)
    finally:
        renderer.shutdown()


def run_training(
    num_episodes: int,
    seed: int,
    max_ticks: int | None = None,
) -> None:
    """Train the shared policy network in headless mode and save weights."""

    resolved_max_ticks = config.MAX_EPISODE_LENGTH if max_ticks is None else max_ticks
    print(f"Mode=train seed={seed} max_ticks={resolved_max_ticks}")
    world = World(max_episode_length=resolved_max_ticks)
    policy_network = create_network_for_world(world, seed=seed)
    trainer = Trainer(world, policy_network, seed=seed)
    trainer.train(num_episodes)
    print(f"Saved policy weights to {config.WEIGHTS_PATH}")


def run_scripted_debug(
    seed: int,
    max_ticks: int | None = None,
    randomize_seed_each_episode: bool = False,
) -> None:
    """Run the existing scripted debug mode with a random inspection network."""

    resolved_max_ticks = config.MAX_EPISODE_LENGTH if max_ticks is None else max_ticks
    print(f"Mode=debug seed={seed} max_ticks={resolved_max_ticks}")
    world = World(max_episode_length=resolved_max_ticks)
    debug_network = create_network_for_world(world, seed=seed)
    episode_seed_rng = random.Random(seed)
    episode_seed_state = {"value": seed}

    def controller_builder(current_world):
        """Build scripted controllers using the current episode seed."""

        return build_scripted_controllers(current_world, episode_seed_state["value"])

    def advance_episode_seed() -> None:
        """Draw a fresh episode seed for the next visible episode."""

        if not randomize_seed_each_episode:
            return
        episode_seed_state["value"] = episode_seed_rng.randint(0, 2**31 - 1)
        print(f"Episode seed: {episode_seed_state['value']}")

    controllers = controller_builder(world)
    run_debug_loop(
        world,
        controllers,
        debug_network,
        controller_builder,
        episode_reset_callback=advance_episode_seed,
    )


def run_policy_debug(
    seed: int,
    max_ticks: int | None = None,
    randomize_seed_each_episode: bool = False,
) -> None:
    """Load saved weights and run the learned policy with visible on-screen learning."""

    resolved_max_ticks = config.MAX_EPISODE_LENGTH if max_ticks is None else max_ticks
    print(f"Mode=policy seed={seed} max_ticks={resolved_max_ticks}")
    world = World(max_episode_length=resolved_max_ticks)
    policy_network = create_network_for_world(world, seed=seed)
    trainer = Trainer(world, policy_network, seed=seed)
    if trainer.load_weights_if_available():
        print(f"Loaded policy weights from {config.WEIGHTS_PATH}")
    else:
        print(
            f"No valid saved weights found at {config.WEIGHTS_PATH}. "
            "Starting from fresh random weights."
        )
    available_modes = PolicyController.available_modes()
    default_mode = (
        config.DEFAULT_POLICY_MODE
        if config.DEFAULT_POLICY_MODE in available_modes
        else available_modes[0]
    )
    policy_mode_state = {"index": available_modes.index(default_mode)}
    episode_seed_rng = random.Random(seed)
    episode_seed_state = {"value": seed}

    def current_policy_mode() -> str:
        """Return the currently selected policy playback mode."""

        return available_modes[policy_mode_state["index"]]

    def cycle_policy_mode() -> None:
        """Advance to the next policy playback mode."""

        policy_mode_state["index"] = (
            policy_mode_state["index"] + 1
        ) % len(available_modes)
        print(f"Policy mode: {current_policy_mode()}")

    def controller_builder(current_world):
        """Build policy controllers using the current episode seed."""

        return build_policy_controllers(
            current_world,
            policy_network,
            episode_seed_state["value"],
            current_policy_mode(),
        )

    def advance_episode_seed() -> None:
        """Draw a fresh episode seed for the next visible episode."""

        if not randomize_seed_each_episode:
            return
        episode_seed_state["value"] = episode_seed_rng.randint(0, 2**31 - 1)
        print(f"Episode seed: {episode_seed_state['value']}")

    controllers = controller_builder(world)
    run_debug_loop(
        world,
        controllers,
        policy_network,
        controller_builder,
        controller_mode_label_getter=current_policy_mode,
        controller_mode_cycle_callback=cycle_policy_mode,
        episode_reset_callback=advance_episode_seed,
        live_trainer=trainer,
    )


def parse_cli_args(
    raw_args: list[str],
) -> tuple[str, list[str], int, int | None, bool]:
    """Parse the main mode arguments and optional shared random seed."""

    filtered_args: list[str] = []
    seed = config.DEFAULT_SEED
    seed_explicitly_set = False
    max_ticks: int | None = None
    index = 0

    while index < len(raw_args):
        current_arg = raw_args[index]
        if current_arg == "--seed":
            if index + 1 >= len(raw_args):
                raise SystemExit("Missing value for --seed")
            seed_arg = raw_args[index + 1]
            seed = random.SystemRandom().randint(0, 2**31 - 1) if seed_arg == "auto" else int(seed_arg)
            seed_explicitly_set = True
            index += 2
            continue
        if current_arg == "--tick":
            if index + 1 >= len(raw_args):
                raise SystemExit("Missing value for --tick")
            max_ticks = int(raw_args[index + 1])
            if max_ticks <= 0:
                raise SystemExit("--tick must be a positive integer")
            index += 2
            continue
        filtered_args.append(current_arg)
        index += 1

    mode = filtered_args[0] if filtered_args else config.DEFAULT_START_MODE
    mode_args = filtered_args[1:] if filtered_args else []
    randomize_seed_each_episode = not raw_args and not seed_explicitly_set
    if not raw_args and not seed_explicitly_set:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    if not raw_args and max_ticks is None:
        max_ticks = config.DEFAULT_START_TICKS
    return mode, mode_args, seed, max_ticks, randomize_seed_each_episode


def main(argv: list[str] | None = None) -> None:
    """Dispatch between scripted debug, training, and learned policy playback."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    mode, mode_args, seed, max_ticks, randomize_seed_each_episode = parse_cli_args(raw_args)

    if mode == "train":
        num_episodes = int(mode_args[0]) if mode_args else config.DEFAULT_TRAIN_EPISODES
        run_training(num_episodes, seed, max_ticks=max_ticks)
    elif mode == "policy":
        run_policy_debug(
            seed,
            max_ticks=max_ticks,
            randomize_seed_each_episode=randomize_seed_each_episode,
        )
    elif mode == "debug":
        run_scripted_debug(
            seed,
            max_ticks=max_ticks,
            randomize_seed_each_episode=randomize_seed_each_episode,
        )
    else:
        raise SystemExit(
            "Usage: python -m survival_ai.main [debug|train [episodes]|policy] [--seed N|auto] [--tick N]"
        )


if __name__ == "__main__":
    main()
