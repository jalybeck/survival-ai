"""Reward calculation for per-step survival, combat, death, and win signals."""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass(slots=True)
class RewardBreakdown:
    """Stores one agent's reward components for a single environment step."""

    total: float = 0.0
    survive_reward: float = 0.0
    damage_reward: float = 0.0
    damage_penalty: float = 0.0
    death_penalty: float = 0.0
    win_reward: float = 0.0
    exploration_reward: float = 0.0
    approach_reward: float = 0.0
    attack_range_reward: float = 0.0
    item_pickup_reward: float = 0.0
    contextual_weapon_pickup_reward: float = 0.0
    contextual_heal_pickup_reward: float = 0.0
    heal_item_use_reward: float = 0.0
    low_health_heal_bonus: float = 0.0
    low_health_retreat_reward: float = 0.0
    low_health_approach_penalty: float = 0.0
    low_health_melee_range_penalty: float = 0.0
    weapon_item_use_reward: float = 0.0
    weapon_hit_bonus: float = 0.0
    weapon_kill_bonus: float = 0.0
    armed_visible_enemy_reward: float = 0.0
    drop_penalty: float = 0.0
    drop_weapon_while_threatened_penalty: float = 0.0
    idle_penalty: float = 0.0
    oscillation_penalty: float = 0.0
    no_winner_penalty: float = 0.0


def create_empty_reward_breakdowns(agent_ids: list[int] | tuple[int, ...] | set[int]) -> dict[int, RewardBreakdown]:
    """Create zero-initialized reward breakdowns for the provided agent ids."""

    return {agent_id: RewardBreakdown() for agent_id in agent_ids}


def compute_step_rewards(world, step_result) -> dict[int, RewardBreakdown]:
    """Calculate one reward breakdown per agent for the latest world step."""

    reward_breakdowns = create_empty_reward_breakdowns(world.agents.keys())

    for agent in world.agents.values():
        if agent.alive:
            reward_breakdowns[agent.entity_id].survive_reward += config.SURVIVE_TICK_REWARD

    for event in step_result.damage_events:
        # Reward is scaled relative to the current melee damage so the default
        # configuration produces the documented +1 / -1 signal per successful hit.
        reward_scale = event.damage / max(1, world.melee_damage)
        reward_breakdowns[event.attacker_id].damage_reward += (
            config.DEAL_DAMAGE_REWARD * reward_scale
        )
        reward_breakdowns[event.target_id].damage_penalty += (
            config.TAKE_DAMAGE_PENALTY * reward_scale
        )
        if event.used_weapon:
            if event.weapon_type == "melee_weapon":
                reward_breakdowns[event.attacker_id].weapon_hit_bonus += (
                    config.MELEE_WEAPON_HIT_REWARD_BONUS
                )
            elif event.weapon_type == "ranged_weapon":
                reward_breakdowns[event.attacker_id].weapon_hit_bonus += (
                    config.RANGED_WEAPON_HIT_REWARD_BONUS
                )
        if event.killed_target and event.used_weapon:
            if event.weapon_type == "melee_weapon":
                reward_breakdowns[event.attacker_id].weapon_kill_bonus += (
                    config.MELEE_WEAPON_KILL_REWARD_BONUS
                )
            elif event.weapon_type == "ranged_weapon":
                reward_breakdowns[event.attacker_id].weapon_kill_bonus += (
                    config.RANGED_WEAPON_KILL_REWARD_BONUS
                )

    for event in step_result.item_events:
        agent_id = int(event["agent_id"])
        agent = world.agents[agent_id]

        if event["event_type"] == "pickup" and int(event.get("reward_granted", 0)) == 1:
            reward_breakdowns[agent_id].item_pickup_reward += (
                config.ITEM_PICKUP_REWARD
            )
            if (
                event["item_type"] in {"melee_weapon", "ranged_weapon"}
                and int(event.get("visible_enemy_at_event", 0)) == 1
            ):
                reward_breakdowns[agent_id].contextual_weapon_pickup_reward += (
                    config.CONTEXTUAL_WEAPON_PICKUP_REWARD
                )
            elif (
                event["item_type"] == "heal"
                and int(event.get("low_health_at_event", 0)) == 1
            ):
                reward_breakdowns[agent_id].contextual_heal_pickup_reward += (
                    config.CONTEXTUAL_HEAL_PICKUP_REWARD
                )
        elif (
            event["event_type"] == "use"
            and event["item_type"] == "heal"
            and int(event["value"]) > 0
        ):
            reward_breakdowns[agent_id].heal_item_use_reward += (
                config.HEAL_ITEM_USE_REWARD
            )
            if (
                agent.used_heal_item_this_step
                and agent.health_before_heal_this_step / max(1, agent.max_health)
                <= config.LOW_HEALTH_THRESHOLD
            ):
                reward_breakdowns[agent_id].low_health_heal_bonus += (
                    config.LOW_HEALTH_HEAL_BONUS
                )
        elif (
            event["event_type"] == "use"
            and event["item_type"] == "weapon"
            and int(event.get("from_drop", 0)) == 0
        ):
            reward_breakdowns[agent_id].weapon_item_use_reward += (
                config.WEAPON_ITEM_USE_REWARD
            )
        elif event["event_type"] == "drop":
            reward_breakdowns[agent_id].drop_penalty += (
                config.DROP_ITEM_PENALTY
            )
            if (
                event["item_type"] in {"melee_weapon", "ranged_weapon"}
                and int(event.get("visible_enemy_at_event", 0)) == 1
            ):
                reward_breakdowns[agent_id].drop_weapon_while_threatened_penalty += (
                    config.DROP_WEAPON_WHILE_THREATENED_PENALTY
                )

    for agent_id in step_result.deaths:
        reward_breakdowns[agent_id].death_penalty += config.DEATH_PENALTY

    if step_result.winner_id is not None:
        reward_breakdowns[step_result.winner_id].win_reward += config.WIN_REWARD
    elif step_result.episode_over:
        for agent in world.alive_agents():
            reward_breakdowns[agent.entity_id].no_winner_penalty += config.NO_WINNER_PENALTY

    for agent in world.agents.values():
        is_low_health = agent.health / max(1, agent.max_health) <= config.LOW_HEALTH_THRESHOLD
        if agent.visited_new_tile_this_step:
            reward_breakdowns[agent.entity_id].exploration_reward += config.NEW_TILE_REWARD
        if (
            agent.previous_visible_enemy_distance is not None
            and agent.current_visible_enemy_distance is not None
        ):
            if agent.current_visible_enemy_distance < agent.previous_visible_enemy_distance:
                if is_low_health:
                    reward_breakdowns[agent.entity_id].low_health_approach_penalty += (
                        config.LOW_HEALTH_APPROACH_PENALTY
                    )
                else:
                    # Reward only actual progress toward a visible enemy. This gives the
                    # policy a dense pursuit signal without rewarding random movement.
                    reward_breakdowns[agent.entity_id].approach_reward += (
                        config.APPROACH_VISIBLE_AGENT_REWARD
                    )
            elif (
                is_low_health
                and agent.current_visible_enemy_distance > agent.previous_visible_enemy_distance
            ):
                reward_breakdowns[agent.entity_id].low_health_retreat_reward += (
                    config.LOW_HEALTH_RETREAT_REWARD
                )
        if agent.entered_attack_range_this_step:
            reward_breakdowns[agent.entity_id].attack_range_reward += (
                config.ENTER_ATTACK_RANGE_REWARD
            )
        if (
            is_low_health
            and agent.current_visible_enemy_distance == 1
        ):
            reward_breakdowns[agent.entity_id].low_health_melee_range_penalty += (
                config.LOW_HEALTH_MELEE_RANGE_PENALTY
            )
        if agent.has_equipped_weapon() and agent.current_visible_enemy_distance is not None:
            reward_breakdowns[agent.entity_id].armed_visible_enemy_reward += (
                config.ARMED_VISIBLE_ENEMY_REWARD
            )
        if agent.idled_last_step:
            reward_breakdowns[agent.entity_id].idle_penalty += config.IDLE_PENALTY
        if agent.oscillated_last_step:
            reward_breakdowns[agent.entity_id].oscillation_penalty += config.OSCILLATION_PENALTY

    for breakdown in reward_breakdowns.values():
        breakdown.total = (
            breakdown.survive_reward
            + breakdown.damage_reward
            + breakdown.damage_penalty
            + breakdown.death_penalty
            + breakdown.win_reward
            + breakdown.exploration_reward
            + breakdown.approach_reward
            + breakdown.attack_range_reward
            + breakdown.item_pickup_reward
            + breakdown.contextual_weapon_pickup_reward
            + breakdown.contextual_heal_pickup_reward
            + breakdown.heal_item_use_reward
            + breakdown.low_health_heal_bonus
            + breakdown.low_health_retreat_reward
            + breakdown.low_health_approach_penalty
            + breakdown.low_health_melee_range_penalty
            + breakdown.weapon_item_use_reward
            + breakdown.weapon_hit_bonus
            + breakdown.weapon_kill_bonus
            + breakdown.armed_visible_enemy_reward
            + breakdown.drop_penalty
            + breakdown.drop_weapon_while_threatened_penalty
            + breakdown.idle_penalty
            + breakdown.oscillation_penalty
            + breakdown.no_winner_penalty
        )

    return reward_breakdowns


def format_reward_breakdown(breakdown: RewardBreakdown) -> str:
    """Format one reward breakdown into a compact debug string."""

    return (
        f"total={breakdown.total:+.2f} "
        f"survive={breakdown.survive_reward:+.2f} "
        f"deal={breakdown.damage_reward:+.2f} "
        f"take={breakdown.damage_penalty:+.2f} "
        f"death={breakdown.death_penalty:+.2f} "
        f"win={breakdown.win_reward:+.2f} "
        f"explore={breakdown.exploration_reward:+.2f} "
        f"approach={breakdown.approach_reward:+.2f} "
        f"range={breakdown.attack_range_reward:+.2f} "
        f"pickup={breakdown.item_pickup_reward:+.2f} "
        f"pickup_weapon_ctx={breakdown.contextual_weapon_pickup_reward:+.2f} "
        f"pickup_heal_ctx={breakdown.contextual_heal_pickup_reward:+.2f} "
        f"heal={breakdown.heal_item_use_reward:+.2f} "
        f"heal_low={breakdown.low_health_heal_bonus:+.2f} "
        f"retreat_low={breakdown.low_health_retreat_reward:+.2f} "
        f"approach_low={breakdown.low_health_approach_penalty:+.2f} "
        f"melee_low={breakdown.low_health_melee_range_penalty:+.2f} "
        f"weapon={breakdown.weapon_item_use_reward:+.2f} "
        f"weapon_hit={breakdown.weapon_hit_bonus:+.2f} "
        f"weapon_kill={breakdown.weapon_kill_bonus:+.2f} "
        f"armed={breakdown.armed_visible_enemy_reward:+.2f} "
        f"drop={breakdown.drop_penalty:+.2f} "
        f"drop_threat={breakdown.drop_weapon_while_threatened_penalty:+.2f} "
        f"idle={breakdown.idle_penalty:+.2f} "
        f"loop={breakdown.oscillation_penalty:+.2f} "
        f"draw={breakdown.no_winner_penalty:+.2f}"
    )
