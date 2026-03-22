"""Map generation helpers for deterministic Phase 1 arena layouts."""

from __future__ import annotations

from dataclasses import dataclass

from .utils import in_bounds


@dataclass(slots=True)
class GridMap:
    """Stores wall coordinates and dimension metadata for the arena."""

    width: int
    height: int
    walls: set[tuple[int, int]]

    def is_wall(self, x: int, y: int) -> bool:
        """Return True when the coordinate contains a wall tile."""

        return (x, y) in self.walls

    def is_inside(self, x: int, y: int) -> bool:
        """Return True when the coordinate is inside the arena bounds."""

        return in_bounds(x, y, self.width, self.height)


def generate_map(width: int, height: int) -> GridMap:
    """Generate a deterministic arena with borders and small obstacles."""

    walls: set[tuple[int, int]] = set()

    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))

    center_x = width // 2
    center_y = height // 2

    interior_blocks = {
        (center_x, center_y),
        (center_x - 1, center_y),
        (center_x + 1, center_y),
        (center_x, center_y - 1),
        (center_x, center_y + 1),
        (3, 3),
        (width - 4, 3),
        (3, height - 4),
        (width - 4, height - 4),
    }
    walls.update(interior_blocks)

    return GridMap(width=width, height=height, walls=walls)
