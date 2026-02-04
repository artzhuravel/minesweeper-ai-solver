"""Utility functions for the Minesweeper AI solver."""

from typing import Dict, List, Tuple

# Module-level cache: (width, height) -> {(x,y): ((nx,ny), ...), ...}
_NEIGHBORHOODS_CACHE: Dict[
    Tuple[int, int],
    Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]
] = {}


def get_neighborhoods(
    width: int, height: int
) -> Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]:
    """
    Precompute and cache 8-connected neighbor coordinates for every cell in a grid.

    Args:
        width: Grid width (number of columns). Must be positive.
        height: Grid height (number of rows). Must be positive.

    Returns:
        Mapping from each cell (x, y) to a tuple of valid neighboring
        coordinates (nx, ny) under 8-connectivity.

    Raises:
        ValueError: If width or height is non-positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    key = (width, height)
    cached = _NEIGHBORHOODS_CACHE.get(key)
    if cached is not None:
        return cached

    neighborhoods: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {}
    for y in range(height):
        for x in range(width):
            nbrs: List[Tuple[int, int]] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        nbrs.append((nx, ny))
            neighborhoods[(x, y)] = tuple(nbrs)

    _NEIGHBORHOODS_CACHE[key] = neighborhoods
    return neighborhoods
