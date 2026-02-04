"""Minesweeper game engine with first-click safety and constrained mine placement."""

import random
from collections import deque
from typing import Deque, Dict, FrozenSet, List, Set, Tuple

from .utils import get_neighborhoods


class Minesweeper:
    """Minesweeper game engine with first-click safety and constrained mine placement."""

    def __init__(
        self,
        width: int,
        height: int,
        mines_count: int,
        mines_generation_algorithm: str,
    ) -> None:
        """
        Initialize a Minesweeper game engine.

        Args:
            width: Board width (number of columns), must be > 0.
            height: Board height (number of rows), must be > 0.
            mines_count: Total number of mines to place, must be >= 0.
            mines_generation_algorithm: Mine placement rule; one of
                {"safe_first_action_rule", "safe_neighborhood_rule"}.

        Raises:
            ValueError: If dimensions are invalid or algorithm is unrecognized.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive.")
        if mines_count < 0:
            raise ValueError("mines_count must be non-negative.")

        self.mines_generation_algorithm: str = mines_generation_algorithm

        if self.mines_generation_algorithm not in (
            "safe_first_action_rule",
            "safe_neighborhood_rule",
        ):
            raise ValueError(
                'mines_generation_algorithm must be "safe_first_action_rule" '
                'or "safe_neighborhood_rule".'
            )

        max_mines = width * height - 9
        if (
            self.mines_generation_algorithm == "safe_first_action_rule"
            and mines_count >= max_mines
        ):
            raise ValueError(
                "Cannot place enough safe cells to satisfy safe_first_action_rule."
            )

        self.width: int = width
        self.height: int = height
        self.mines_count: int = mines_count

        self.board: List[List[str]] = [
            [" " for _ in range(width)] for _ in range(height)
        ]
        self.board_blank: bool = True
        self.revealed: List[List[bool]] = [
            [False for _ in range(width)] for _ in range(height)
        ]
        self.first_move: bool = True

        self.unrevealed_count: int = width * height - mines_count
        self.game_over: bool = False

        self._neighborhoods: Dict[
            Tuple[int, int], Tuple[Tuple[int, int], ...]
        ] = get_neighborhoods(width, height)

    def reset(self) -> None:
        """
        Reset the game state to allow re-solving the same board.

        Keeps the mine positions but resets all revealed cells and game state.
        """
        self.revealed = [
            [False for _ in range(self.width)] for _ in range(self.height)
        ]
        self.unrevealed_count = self.width * self.height - self.mines_count
        self.game_over = False
        # Don't reset first_move or board_blank - mines are already placed

    def neighbors(self, x: int, y: int) -> Tuple[Tuple[int, int], ...]:
        """
        Return precomputed neighbor coordinates for a cell.

        Args:
            x: Cell x-coordinate (column).
            y: Cell y-coordinate (row).

        Returns:
            All valid (nx, ny) neighbors in the 8-neighborhood.
        """
        return self._neighborhoods[(x, y)]

    def place_mines(self, first_x: int, first_y: int) -> None:
        """
        Place mines on the board (one-time), respecting the selected first-move safety rule.

        Args:
            first_x: X-coordinate of the first revealed cell.
            first_y: Y-coordinate of the first revealed cell.

        Raises:
            ValueError: If the board is not blank or mines cannot be placed.
        """
        if not self.board_blank:
            raise ValueError("The board is not blank.")

        if self.mines_generation_algorithm == "safe_first_action_rule":
            if self.mines_count > self.width * self.height - 1:
                raise ValueError(
                    "Cannot place enough safe cells to satisfy safe_first_action_rule."
                )

            # Only the first clicked cell is guaranteed safe.
            safe: Set[Tuple[int, int]] = {(first_x, first_y)}

            eligible: List[Tuple[int, int]] = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) not in safe
            ]

            mines = set(random.sample(eligible, self.mines_count))
            for mx, my in mines:
                self.board[my][mx] = "M"

            self.board_blank = False

        else:  # "safe_neighborhood_rule"
            if self.mines_count > self.width * self.height - 9:
                raise ValueError(
                    "Cannot place enough safe cells to satisfy safe_neighborhood_rule."
                )

            # Safe zone = first click + its neighbors.
            safe: Set[Tuple[int, int]] = set(
                self.neighbors(first_x, first_y)
            ) | {(first_x, first_y)}

            # Eligible cells = all cells not in the safe zone.
            eligible: List[Tuple[int, int]] = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) not in safe
            ]

            # Sample mines uniformly without replacement.
            mines = set(random.sample(eligible, self.mines_count))
            for mx, my in mines:
                self.board[my][mx] = "M"

            self.board_blank = False

    def get_adjacent_mine_counts(self) -> None:
        """Populate every non-mine cell with its adjacent mine count."""
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == "M":
                    continue

                count = sum(
                    1 for nx, ny in self.neighbors(x, y) if self.board[ny][nx] == "M"
                )
                self.board[y][x] = str(count)

    def flood_fill(self, x: int, y: int) -> List[Tuple[int, int, str]]:
        """
        Reveal a connected region starting at (x, y) using Minesweeper flood fill rules.

        Args:
            x: X-coordinate of the starting cell.
            y: Y-coordinate of the starting cell.

        Returns:
            A list of newly revealed cells as (x, y, value_str).
        """
        frontier: Deque[Tuple[int, int]] = deque([(x, y)])
        visited: Set[Tuple[int, int]] = {(x, y)}
        revealed_cells: List[Tuple[int, int, str]] = []

        while frontier:
            cx, cy = frontier.popleft()
            if self.revealed[cy][cx]:
                continue

            self.revealed[cy][cx] = True
            self.unrevealed_count -= 1
            revealed_cells.append((cx, cy, self.board[cy][cx]))

            if self.board[cy][cx] == "0":
                for nx, ny in self.neighbors(cx, cy):
                    if (nx, ny) in visited or self.revealed[ny][nx]:
                        continue
                    visited.add((nx, ny))
                    frontier.append((nx, ny))

        return revealed_cells

    def reveal(self, x: int, y: int) -> Tuple[int, Dict[str, object]]:
        """
        Reveal a single cell and return a status code plus payload.

        Args:
            x: X-coordinate of the cell to reveal.
            y: Y-coordinate of the cell to reveal.

        Returns:
            Tuple of (status, payload) where status is:
                - -1: Mine hit (loss)
                - 0: Non-terminal reveal (or no-op)
                - 1: Win (all safe cells revealed)

            Payload contains:
                - For status 0 or 1: {"revealed_cells": List[(x, y, value_str)]}
                - For status -1: {"revealed_cells_count": int, "all_mines": FrozenSet}

        Raises:
            ValueError: If coordinates are out of bounds.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise ValueError("Cell coordinates are outside the board.")

        if self.game_over:
            return 0, {}

        if self.revealed[y][x]:
            return 0, {}

        if self.first_move:
            self.place_mines(x, y)
            self.get_adjacent_mine_counts()
            self.first_move = False

        if self.board[y][x] == "M":
            self.revealed[y][x] = True
            self.game_over = True

            revealed_cells_count = (
                self.width * self.height - self.mines_count
            ) - self.unrevealed_count
            all_mines: FrozenSet[Tuple[int, int]] = frozenset(
                (cx, cy)
                for cy in range(self.height)
                for cx in range(self.width)
                if self.board[cy][cx] == "M"
            )
            return -1, {
                "revealed_cells_count": revealed_cells_count,
                "all_mines": all_mines,
            }

        revealed_cells = self.flood_fill(x, y)

        if self.unrevealed_count == 0:
            self.game_over = True
            return 1, {"revealed_cells": revealed_cells}

        return 0, {"revealed_cells": revealed_cells}

    # -------------------------------------------------------------------------
    # Display methods
    # -------------------------------------------------------------------------

    _ANSI_RESET = "\033[0m"
    _ANSI_COORD = "\033[96m"
    _ANSI_MINE = "\033[91m"

    def _c(self, s: str) -> str:
        """Wrap string in coordinate color."""
        return f"{self._ANSI_COORD}{s}{self._ANSI_RESET}"

    def _m(self, s: str) -> str:
        """Wrap string in mine color (red)."""
        return f"{self._ANSI_MINE}{s}{self._ANSI_RESET}"

    def format_board(self, reveal_all: bool = False) -> str:
        """
        Render the board as a multi-line string for terminal display.

        Args:
            reveal_all: If True, show mines and all underlying values.

        Returns:
            A formatted multi-line string with coordinate labels and the board grid.
        """
        w, h = self.width, self.height

        def cell_str(x: int, y: int) -> str:
            if reveal_all or self.revealed[y][x]:
                v = self.board[y][x]
                if v == "M":
                    return self._m("M")
                return v
            return "."

        # Header: x coordinates
        header_cells = " ".join(f"{x:2d}" for x in range(w))
        out = [self._c("   ") + self._c(header_cells)]

        # Separator line
        sep = self._c("   " + "-" * (3 * w - 1))
        out.append(sep)

        # Rows with y coordinate at left
        for y in range(h):
            row_cells = " ".join(f" {cell_str(x, y)}" for x in range(w))
            out.append(self._c(f"{y:2d} ") + self._c("|") + row_cells)

        return "\n".join(out)

    def print_board(self) -> None:
        """Print the current visible board state to stdout."""
        print(self.format_board(reveal_all=False))

    def print_full_board(self) -> None:
        """Print the fully revealed underlying board to stdout (for debugging)."""
        print(self.format_board(reveal_all=True))


def play_cli(game: Minesweeper) -> None:
    """
    Run a simple terminal UI for playing Minesweeper.

    Args:
        game: A Minesweeper instance to play against.
    """
    print("Minesweeper CLI (enter: x y). Coordinates are 0-based. Type 'q' to quit.\n")
    print(game.format_board(reveal_all=False))

    while True:
        s = input("\nMove (x y): ").strip()
        if s.lower() in {"q", "quit", "exit"}:
            print("Quit.")
            return

        parts = s.replace(",", " ").split()
        if len(parts) != 2:
            print("Invalid input. Example: 3 5")
            continue

        try:
            x = int(parts[0])
            y = int(parts[1])
        except ValueError:
            print("Invalid input. Coordinates must be integers.")
            continue

        status, _ = game.reveal(x, y)

        print(f"\nYou decided to reveal ({x}, {y}).\n")
        print(game.format_board(reveal_all=False))

        if status == -1:
            print("\nYou hit a mine. You lost.")
            print("\nFull board:")
            print(game.format_board(reveal_all=True))
            return

        if status == 1:
            print("\nYou revealed all safe cells. You won!")
            print("\nFull board:")
            print(game.format_board(reveal_all=True))
            return
