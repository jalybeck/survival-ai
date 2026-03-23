"""Pygame-based rendering for the grid world and observation debugging overlays."""

from __future__ import annotations

import os

from . import config
try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover - depends on local environment.
    pygame = None


class PygameRenderer:
    """Draws the current world state to a pygame window."""

    def __init__(self, width: int, height: int) -> None:
        if pygame is None:
            raise RuntimeError("pygame is not installed. Install it with `pip install pygame`.")

        os.environ["SDL_VIDEO_CENTERED"] = "1"
        pygame.init()
        self.cell_size = config.CELL_SIZE
        self.top_panel_height = config.TOP_PANEL_HEIGHT
        self.bottom_panel_height = config.BOTTOM_PANEL_HEIGHT
        self.map_pixel_width = width * self.cell_size
        self.map_pixel_height = height * self.cell_size
        self.base_window_width = self.map_pixel_width + config.WINDOW_EXTRA_WIDTH
        self.network_panel_width = max(config.NETWORK_PANEL_WIDTH, self.base_window_width)
        self.window_height = self.top_panel_height + self.map_pixel_height + self.bottom_panel_height
        self.network_panel_expanded = True
        self.window_width = self.base_window_width + self.network_panel_width
        self.map_x_offset = (self.base_window_width - self.map_pixel_width) // 2
        self.surface = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Survival AI - Phase 6")
        self._font = pygame.font.SysFont("consolas", 20)
        self._small_font = pygame.font.SysFont("consolas", 16)
        self._tiny_font = pygame.font.SysFont("consolas", 15)
        self._micro_font = pygame.font.SysFont("consolas", 13)
        self._clock = pygame.time.Clock()

    @property
    def clock(self):
        """Expose the internal frame limiter."""

        return self._clock

    def draw(
        self,
        world,
        visible_cells: set[tuple[int, int]] | None = None,
        observation_snapshot=None,
        previous_observation_snapshot=None,
        debug_agent_id: int | None = None,
        auto_advance: bool = False,
        controller_mode_label: str | None = None,
        training_active_label: str | None = None,
        debug_agent_dead: bool = False,
        tracer_lines: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]] | None = None,
        network_trace=None,
        previous_network_trace=None,
        action_labels: list[str] | None = None,
        network_debug_lines: list[str] | None = None,
    ) -> None:
        """Render the world state and selected-agent observation debug data."""

        self.surface.fill(config.BACKGROUND_COLOR)
        self._draw_top_panel(world, auto_advance, controller_mode_label, training_active_label)
        self._draw_map(world, visible_cells or set())
        self._draw_tracers(tracer_lines or [])
        self._draw_agents(world)
        self._draw_bottom_panel(
            world,
            observation_snapshot,
            debug_agent_id,
            debug_agent_dead,
            network_debug_lines or [],
        )
        self._draw_network_panel(
            observation_snapshot,
            previous_observation_snapshot,
            network_trace,
            previous_network_trace,
            action_labels or [],
            debug_agent_dead,
        )
        pygame.display.flip()

    def handle_mouse_click(self, mouse_position: tuple[int, int]) -> bool:
        """Toggle the network panel when the user clicks the top-right button."""

        if not self._network_button_rect().collidepoint(mouse_position):
            return False
        self.network_panel_expanded = not self.network_panel_expanded
        self.window_width = (
            self.base_window_width + self.network_panel_width
            if self.network_panel_expanded
            else self.base_window_width
        )
        self.surface = pygame.display.set_mode((self.window_width, self.window_height))
        return True

    def shutdown(self) -> None:
        """Close pygame and release renderer resources."""

        if pygame is not None:
            pygame.quit()

    def _draw_map(self, world, visible_cells: set[tuple[int, int]]) -> None:
        """Draw the arena floor, walls, and optional visible-cell overlay."""

        overlay_surface = pygame.Surface(
            (world.width * self.cell_size, world.height * self.cell_size),
            pygame.SRCALPHA,
        )

        for y in range(world.height):
            for x in range(world.width):
                rect = pygame.Rect(
                    self.map_x_offset + x * self.cell_size,
                    self.top_panel_height + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                color = config.WALL_COLOR if world.grid_map.is_wall(x, y) else config.FLOOR_COLOR
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, config.GRID_LINE_COLOR, rect, 1)

                if (x, y) in visible_cells:
                    overlay_rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(overlay_surface, config.VISIBLE_OVERLAY_COLOR, overlay_rect)

        self.surface.blit(overlay_surface, (self.map_x_offset, self.top_panel_height))

    def _draw_agents(self, world) -> None:
        """Draw living and dead agents with simple health labels."""

        for item in world.items.values():
            if not item.alive:
                continue
            center_x = self.map_x_offset + item.x * self.cell_size + self.cell_size // 2
            center_y = self.top_panel_height + item.y * self.cell_size + self.cell_size // 2
            if item.item_type == "heal":
                color = (87, 190, 112)
                label_text = "H"
            elif item.item_type == "melee_weapon":
                color = (222, 170, 64)
                label_text = "M"
            else:
                color = (92, 164, 230)
                label_text = "R"
            pygame.draw.rect(
                self.surface,
                color,
                pygame.Rect(center_x - 10, center_y - 10, 20, 20),
                border_radius=4,
            )
            label = self._tiny_font.render(label_text, True, (12, 14, 18))
            label_rect = label.get_rect(center=(center_x, center_y))
            self.surface.blit(label, label_rect)

        # Draw dead agents first so a living agent standing on the same tile
        # remains visible after moving onto a defeated opponent's position.
        for agent in sorted(world.agents.values(), key=lambda item: item.alive):
            color = config.AGENT_COLORS[(agent.entity_id - 1) % len(config.AGENT_COLORS)]
            center_x = self.map_x_offset + agent.x * self.cell_size + self.cell_size // 2
            center_y = self.top_panel_height + agent.y * self.cell_size + self.cell_size // 2

            if not agent.alive:
                faded_color = tuple(max(40, channel // 3) for channel in color)
                pygame.draw.circle(
                    self.surface,
                    faded_color,
                    (center_x, center_y),
                    self.cell_size // 5,
                    width=2,
                )
            else:
                pygame.draw.circle(self.surface, color, (center_x, center_y), self.cell_size // 3)

            hp_label = self._small_font.render(str(agent.health), True, config.TEXT_COLOR)
            hp_rect = hp_label.get_rect(center=(center_x, center_y - self.cell_size // 2 + 12))
            self.surface.blit(hp_label, hp_rect)

    def _draw_tracers(
        self,
        tracer_lines: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]],
    ) -> None:
        """Draw short-lived shot tracer lines on top of the map."""

        for start_cell, end_cell, color in tracer_lines:
            start_x = self.map_x_offset + start_cell[0] * self.cell_size + self.cell_size // 2
            start_y = self.top_panel_height + start_cell[1] * self.cell_size + self.cell_size // 2
            end_x = self.map_x_offset + end_cell[0] * self.cell_size + self.cell_size // 2
            end_y = self.top_panel_height + end_cell[1] * self.cell_size + self.cell_size // 2
            pygame.draw.line(self.surface, color, (start_x, start_y), (end_x, end_y), width=4)
            pygame.draw.circle(self.surface, color, (end_x, end_y), 5)

    def _draw_top_panel(
        self,
        world,
        auto_advance: bool,
        controller_mode_label: str | None,
        training_active_label: str | None,
    ) -> None:
        """Draw the global tick, controls, and agent summaries above the game board."""

        panel_rect = pygame.Rect(0, 0, self.map_pixel_width, self.top_panel_height)
        panel_rect.width = self.window_width
        pygame.draw.rect(self.surface, (14, 16, 20), panel_rect)

        mode_label = "AUTO" if auto_advance else "MANUAL"
        headline_left = (
            f"Tick {world.tick}/{world.max_episode_length}  |  "
            f"Alive: {len(world.alive_agents())}  |  Mode: {mode_label}"
        )
        if controller_mode_label is not None:
            headline_left += f"  |  Policy: {controller_mode_label.upper()}"
        headline_label = self._small_font.render(headline_left, True, config.TEXT_COLOR)
        self.surface.blit(headline_label, (12, 10))

        if training_active_label is not None:
            training_color = (
                (88, 214, 141)
                if training_active_label == "ON"
                else (235, 102, 102)
            )
            training_prefix = self._small_font.render("Training:", True, config.TEXT_COLOR)
            prefix_rect = training_prefix.get_rect(topright=(self.base_window_width - 110, 10))
            self.surface.blit(training_prefix, prefix_rect)
            training_value = self._small_font.render(training_active_label, True, training_color)
            value_rect = training_value.get_rect(topleft=(prefix_rect.right + 8, 10))
            self.surface.blit(training_value, value_rect)

        controls = "Space: step  A: auto  R: reset  Tab: debug agent  O: print obs  Click >>: panel"
        if controller_mode_label is not None:
            controls += "  M: policy mode"
        controls_label = self._small_font.render(controls, True, config.TEXT_COLOR)
        self.surface.blit(controls_label, (12, 38))

        button_rect = self._network_button_rect()
        pygame.draw.rect(self.surface, (38, 42, 50), button_rect, border_radius=4)
        pygame.draw.rect(self.surface, config.NETWORK_PANEL_BORDER, button_rect, width=1, border_radius=4)
        button_text = "<<" if self.network_panel_expanded else ">>"
        button_label = self._small_font.render(button_text, True, config.TEXT_COLOR)
        button_label_rect = button_label.get_rect(center=button_rect.center)
        self.surface.blit(button_label, button_label_rect)

        active_agents = [agent for agent in world.agents.values() if agent.alive]
        status_title = self._small_font.render("Active Agents", True, config.TEXT_COLOR)
        self.surface.blit(status_title, (12, 64))
        item_legend_title = self._small_font.render("Items", True, config.TEXT_COLOR)
        self.surface.blit(item_legend_title, (470, 64))
        self._draw_item_legend(470, 86)

        if not active_agents:
            no_agents_label = self._tiny_font.render("None", True, config.TEXT_COLOR)
            self.surface.blit(no_agents_label, (12, 86))
            return

        for index, agent in enumerate(active_agents):
            color = config.AGENT_COLORS[(agent.entity_id - 1) % len(config.AGENT_COLORS)]
            text = (
                f"A{agent.entity_id}  HP {agent.health:>2}  "
                f"K {agent.kills}  D {agent.damage_dealt}  T {agent.damage_taken}  "
                f"I {agent.inventory_item_type or '-'}  "
                f"E {agent.equipped_weapon_type or '-'}:{agent.equipped_weapon_charges}  "
                f"R{agent.last_reward:+.2f}/{agent.total_reward:+.2f}"
            )
            label = self._tiny_font.render(text, True, color)
            self.surface.blit(label, (12, 86 + index * 16))

    def _draw_bottom_panel(
        self,
        world,
        observation_snapshot,
        debug_agent_id: int | None,
        debug_agent_dead: bool,
        network_debug_lines: list[str],
    ) -> None:
        """Draw the selected-agent observation and network debug panels below the game board."""

        panel_top = self.top_panel_height + self.map_pixel_height
        panel_rect = pygame.Rect(0, panel_top, self.window_width, self.bottom_panel_height)
        pygame.draw.rect(self.surface, (14, 16, 20), panel_rect)

        agent_column_x = 12
        grid_column_x = 250
        feature_column_x = 470
        feature_second_column_x = 620
        network_column_x = 790
        column_top = panel_top + 16
        row_spacing = 16

        agent_title = self._small_font.render("Agents / Debug", True, config.TEXT_COLOR)
        self.surface.blit(agent_title, (agent_column_x, column_top))

        if observation_snapshot is None or debug_agent_id is None:
            return

        debug_agent_status = " DEAD" if debug_agent_dead else ""
        debug_lines = [
            f"Debug Agent A{debug_agent_id}{debug_agent_status}",
            f"Nearest: {observation_snapshot.nearest_visible_agent_id}",
            f"dx={observation_snapshot.nearest_visible_agent_dx}  dy={observation_snapshot.nearest_visible_agent_dy}",
            f"dist={observation_snapshot.nearest_visible_agent_distance}",
            f"reward={world.agents[debug_agent_id].last_reward:+.2f}",
            f"episode_total={world.agents[debug_agent_id].total_reward:+.2f}",
            f"last_action={world.agents[debug_agent_id].last_action.name}",
        ]
        for row_index, line in enumerate(debug_lines):
            debug_label = self._tiny_font.render(line, True, config.TEXT_COLOR)
            self.surface.blit(debug_label, (agent_column_x, column_top + 92 + row_index * row_spacing))

        observation_title = self._small_font.render("Local Observation Grid", True, config.TEXT_COLOR)
        self.surface.blit(observation_title, (grid_column_x, column_top))

        for row_index, row_text in enumerate(observation_snapshot.local_grid_lines):
            row_label = self._tiny_font.render(row_text, True, config.TEXT_COLOR)
            self.surface.blit(row_label, (grid_column_x, column_top + 24 + row_index * 18))

        feature_title = self._small_font.render("Feature Summary", True, config.TEXT_COLOR)
        self.surface.blit(feature_title, (feature_column_x, column_top))

        feature_lines = self._summarize_features(observation_snapshot)
        split_index = (len(feature_lines) + 1) // 2
        for row_index, line in enumerate(feature_lines[:split_index]):
            feature_label = self._tiny_font.render(line, True, config.TEXT_COLOR)
            self.surface.blit(feature_label, (feature_column_x, column_top + 24 + row_index * row_spacing))
        for row_index, line in enumerate(feature_lines[split_index:]):
            feature_label = self._tiny_font.render(line, True, config.TEXT_COLOR)
            self.surface.blit(
                feature_label,
                (feature_second_column_x, column_top + 24 + row_index * row_spacing),
            )

        network_title = self._small_font.render("Network Scores", True, config.TEXT_COLOR)
        self.surface.blit(network_title, (network_column_x, column_top))

        for row_index, line in enumerate(network_debug_lines):
            score_label = self._tiny_font.render(line, True, config.TEXT_COLOR)
            self.surface.blit(score_label, (network_column_x, column_top + 24 + row_index * row_spacing))

    def _draw_network_panel(
        self,
        observation_snapshot,
        previous_observation_snapshot,
        network_trace,
        previous_network_trace,
        action_labels: list[str],
        debug_agent_dead: bool,
    ) -> None:
        """Draw a full right-side network inspection panel when expanded."""

        if (
            not self.network_panel_expanded
            or observation_snapshot is None
            or network_trace is None
        ):
            return

        panel_left = self.base_window_width
        panel_rect = pygame.Rect(
            panel_left,
            0,
            self.window_width - self.base_window_width,
            self.window_height,
        )
        pygame.draw.rect(self.surface, config.NETWORK_PANEL_BACKGROUND, panel_rect)
        pygame.draw.line(
            self.surface,
            config.NETWORK_PANEL_BORDER,
            (panel_left, 0),
            (panel_left, self.window_height),
            width=2,
        )

        title_text = f"Agent A{observation_snapshot.agent_id}"
        if debug_agent_dead:
            title_text += " DEAD"
        title_text += " Network Perspective"
        title = self._font.render(title_text, True, config.TEXT_COLOR)
        self.surface.blit(title, (panel_left + 16, 12))
        legend_y = 34
        legend_label = self._small_font.render("Legend:", True, config.TEXT_COLOR)
        self.surface.blit(legend_label, (panel_left + 16, legend_y))
        increase_label = self._small_font.render("increase", True, (88, 214, 141))
        self.surface.blit(increase_label, (panel_left + 96, legend_y))
        separator_label = self._small_font.render("|", True, config.TEXT_COLOR)
        self.surface.blit(separator_label, (panel_left + 176, legend_y))
        decrease_label = self._small_font.render("decrease", True, (235, 102, 102))
        self.surface.blit(decrease_label, (panel_left + 192, legend_y))
        since_label = self._small_font.render("since previous tick", True, config.TEXT_COLOR)
        self.surface.blit(since_label, (panel_left + 284, legend_y))

        hidden_layer_values = network_trace.activations[1:-1]
        previous_hidden_layer_values = (
            previous_network_trace.activations[1:-1]
            if previous_network_trace is not None
            else []
        )
        column_titles = ["Input"]
        columns = [
            self._format_input_lines(
                observation_snapshot.feature_names,
                network_trace.inputs,
                (
                    previous_observation_snapshot.feature_vector
                    if previous_observation_snapshot is not None
                    else None
                ),
            ),
        ]
        for layer_index, layer_values in enumerate(hidden_layer_values, start=1):
            previous_values = (
                previous_hidden_layer_values[layer_index - 1]
                if layer_index - 1 < len(previous_hidden_layer_values)
                else None
            )
            column_titles.append(f"Hidden{layer_index}")
            columns.append(
                self._format_layer_lines(
                    f"H{layer_index}",
                    layer_values,
                    previous_values,
                )
            )
        column_titles.extend(["Output", "Actions"])
        columns.extend(
            [
                self._format_layer_lines(
                    "O",
                    network_trace.outputs,
                    (
                        previous_network_trace.outputs
                        if previous_network_trace is not None
                        else None
                    ),
                ),
                self._format_action_lines(
                    action_labels,
                    network_trace.outputs,
                    (
                        previous_network_trace.outputs
                        if previous_network_trace is not None
                        else None
                    ),
                ),
            ]
        )
        column_width = max(150, (panel_rect.width - 36) // len(column_titles))
        column_top = 64

        for column_index, (title_text, lines) in enumerate(zip(column_titles, columns, strict=False)):
            column_x = panel_left + 16 + column_index * column_width
            title_label = self._small_font.render(title_text, True, config.TEXT_COLOR)
            self.surface.blit(title_label, (column_x, column_top))
            for row_index, (line, line_color) in enumerate(lines):
                row_label = self._micro_font.render(line, True, line_color)
                self.surface.blit(row_label, (column_x, column_top + 22 + row_index * 14))

    def _summarize_features(self, observation_snapshot) -> list[str]:
        """Build a compact HUD summary while the full vector stays printable via the terminal."""

        feature_lookup = dict(
            zip(
                observation_snapshot.feature_names,
                observation_snapshot.feature_vector,
                strict=False,
            )
        )
        return [
            f"self_x={feature_lookup['self_x_norm']:.2f}",
            f"self_y={feature_lookup['self_y_norm']:.2f}",
            f"health={feature_lookup['self_health_norm']:.2f}",
            f"low_health={feature_lookup.get('low_health_state', 0.0):.2f}",
            (
                "walls="
                f"{int(feature_lookup['wall_up'])}"
                f"{int(feature_lookup['wall_down'])}"
                f"{int(feature_lookup['wall_left'])}"
                f"{int(feature_lookup['wall_right'])}"
            ),
            (
                "adj_agents="
                f"{int(feature_lookup['visible_agent_up'])}"
                f"{int(feature_lookup['visible_agent_down'])}"
                f"{int(feature_lookup['visible_agent_left'])}"
                f"{int(feature_lookup['visible_agent_right'])}"
            ),
            f"nearest_dx={feature_lookup['nearest_agent_dx_norm']:.2f}",
            f"nearest_dy={feature_lookup['nearest_agent_dy_norm']:.2f}",
            f"nearest_dist={feature_lookup['nearest_agent_distance_norm']:.2f}",
            f"enemy_visible={feature_lookup.get('visible_enemy_exists', 0.0):.2f}",
            f"damaged={feature_lookup['damaged_last_step']:.2f}",
            f"can_attack={feature_lookup['can_attack']:.2f}",
            f"in_melee={feature_lookup.get('visible_enemy_in_melee_range', 0.0):.2f}",
            f"can_shoot={feature_lookup['can_shoot']:.2f}",
            f"in_shoot_line={feature_lookup.get('visible_enemy_in_shoot_line', 0.0):.2f}",
            f"item_here={feature_lookup['item_here']:.2f}",
            (
                "tile_item="
                f"{int(feature_lookup.get('standing_on_heal_item', 0.0))}"
                f"{int(feature_lookup.get('standing_on_melee_weapon_item', 0.0))}"
                f"{int(feature_lookup.get('standing_on_ranged_weapon_item', 0.0))}"
            ),
            f"has_item={feature_lookup['has_item']:.2f}",
            f"melee_weapon={feature_lookup.get('equipped_melee_weapon', feature_lookup['has_melee_weapon']):.2f}",
            f"ranged_weapon={feature_lookup.get('equipped_ranged_weapon', feature_lookup['has_ranged_weapon']):.2f}",
            f"weapon_charges={feature_lookup.get('equipped_weapon_charges_norm', 0.0):.2f}",
            f"nearest_item={feature_lookup['nearest_item_distance_norm']:.2f}",
            f"visible_agents={feature_lookup['visible_agent_count_norm']:.2f}",
            f"visible_items={feature_lookup['visible_item_count_norm']:.2f}",
            f"visible_cells={feature_lookup['visible_cell_ratio']:.2f}",
            "Press O for full vector",
        ]

    def _network_button_rect(self):
        """Return the clickable toggle button rect for the network inspection panel."""

        return pygame.Rect(
            self.base_window_width - config.NETWORK_PANEL_BUTTON_WIDTH - 12,
            10,
            config.NETWORK_PANEL_BUTTON_WIDTH,
            config.NETWORK_PANEL_BUTTON_HEIGHT,
        )

    def _draw_item_legend(self, origin_x: int, origin_y: int) -> None:
        """Draw a compact legend for the three item marker types."""

        legend_entries = [
            ("H", "Heal", (87, 190, 112)),
            ("M", "Melee", (222, 170, 64)),
            ("R", "Ranged", (92, 164, 230)),
        ]
        for index, (marker, label_text, color) in enumerate(legend_entries):
            entry_x = origin_x + index * 120
            marker_rect = pygame.Rect(entry_x, origin_y, 18, 18)
            pygame.draw.rect(self.surface, color, marker_rect, border_radius=4)
            marker_label = self._tiny_font.render(marker, True, (12, 14, 18))
            marker_label_rect = marker_label.get_rect(center=marker_rect.center)
            self.surface.blit(marker_label, marker_label_rect)
            text_label = self._tiny_font.render(label_text, True, config.TEXT_COLOR)
            self.surface.blit(text_label, (entry_x + 26, origin_y + 1))

    @staticmethod
    def _short_feature_name(feature_name: str, max_length: int = 13) -> str:
        """Shorten feature names so the right-side panel remains readable."""

        if len(feature_name) <= max_length:
            return feature_name
        return feature_name[: max_length - 1] + "…"

    @staticmethod
    def _delta_color(current_value: float, previous_value: float | None) -> tuple[int, int, int]:
        """Color values by directional change since the previous simulation tick."""

        if previous_value is None:
            return config.TEXT_COLOR
        delta = current_value - previous_value
        if delta > 1e-6:
            return (88, 214, 141)
        if delta < -1e-6:
            return (235, 102, 102)
        return config.TEXT_COLOR

    def _format_input_lines(
        self,
        feature_names: list[str],
        values: list[float],
        previous_values: list[float] | None,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Format the input feature column for the network perspective panel."""

        return [
            (
                f"{index:02d} {self._short_feature_name(name):<13} {value:+.3f}",
                self._delta_color(
                    value,
                    previous_values[index] if previous_values is not None else None,
                ),
            )
            for index, (name, value) in enumerate(zip(feature_names, values, strict=False))
        ]

    def _format_layer_lines(
        self,
        prefix: str,
        values: list[float],
        previous_values: list[float] | None,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Format one hidden or output layer column."""

        return [
            (
                f"{prefix}{index:02d} {value:+.3f}",
                self._delta_color(
                    value,
                    previous_values[index] if previous_values is not None else None,
                ),
            )
            for index, value in enumerate(values)
        ]

    def _format_action_lines(
        self,
        action_labels: list[str],
        output_values: list[float],
        previous_output_values: list[float] | None,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Format the action-name column aligned with output neuron scores."""

        return [
            (
                f"{label:<16} {value:+.3f}",
                self._delta_color(
                    value,
                    previous_output_values[index] if previous_output_values is not None else None,
                ),
            )
            for index, (label, value) in enumerate(zip(action_labels, output_values, strict=False))
        ]
