"""Shared utility helpers used across the simulation modules."""

from __future__ import annotations

from itertools import count


def in_bounds(x: int, y: int, width: int, height: int) -> bool:
    """Return True when a coordinate is inside the map rectangle."""

    return 0 <= x < width and 0 <= y < height


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return the Manhattan distance between two grid coordinates."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def sign(value: int) -> int:
    """Return the sign of an integer as -1, 0, or 1."""

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def create_id_counter(start: int = 1):
    """Create a simple incremental integer identifier generator."""

    return count(start=start)
