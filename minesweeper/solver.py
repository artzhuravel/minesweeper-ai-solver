"""Minesweeper AI solver using constraint-based inference and probabilistic guessing."""

import itertools
from collections import defaultdict, deque
from math import comb
from typing import (
    AbstractSet,
    Any,
    DefaultDict,
    Deque,
    Dict,
    FrozenSet,
    List,
    Set,
    Tuple,
    Union,
    cast,
)

from .engine import Minesweeper
from .utils import get_neighborhoods


class MinesweeperSolver:
    """
    Constraint-based Minesweeper solver with multiple inference strategies.

    The solver uses a tiered approach:
    1. Single inference: Trivial constraint propagation
    2. Paired inference: Overlap-based deduction between constraint pairs
    3. SAT/DFS inference: Exhaustive enumeration of satisfying assignments
    4. Probabilistic guessing: Bayesian estimation when no forced moves exist
    """

    def __init__(
        self,
        game: Minesweeper,
        max_dfs_subgroup_len: Union[int, float] = float("inf"),
        record_steps: bool = True,
        guessing_strategy: str = "bayesian",
    ) -> None:
        """
        Initialize a Minesweeper solving agent bound to a specific game instance.

        Args:
            game: The Minesweeper engine instance to interact with.
            max_dfs_subgroup_len: Maximum subgroup size allowed for SAT/DFS inference;
                larger connected components are skipped to control runtime.
            record_steps: If True, record step history for replay functionality.
                Set to False for benchmarks to improve performance.
            guessing_strategy: Strategy for probabilistic guessing when no forced moves exist.
                "bayesian" (default): Use configuration-based Bayesian estimation when available.
                "local_density": Always use local constraint density approximation.
        """
        if guessing_strategy not in ("bayesian", "local_density"):
            raise ValueError(
                'guessing_strategy must be "bayesian" or "local_density".'
            )
        self.game = game
        self.record_steps = record_steps
        self.guessing_strategy = guessing_strategy
        self.board_height: int = game.height
        self.board_width: int = game.width
        self.mines_generation_algorithm: str = game.mines_generation_algorithm

        # Cached 8-neighborhoods: (x,y) -> ((nx,ny), ...)
        self._neighborhoods: Dict[
            Tuple[int, int], Tuple[Tuple[int, int], ...]
        ] = get_neighborhoods(self.board_width, self.board_height)

        # Global bookkeeping
        self.unknown_count: int = self.board_height * self.board_width
        self.expected_mines_count: int = game.mines_count

        # Metrics / counters (for analysis)
        self.reveal_moves_count: int = 0
        self.max_revealed_frontier: int = 0
        self.max_unrevealed_frontier: int = 0
        self.inferred_single_count: int = 0
        self.attempted_single_count: int = 0
        self.inferred_paired_count: int = 0
        self.attempted_paired_count: int = 0
        self.inferred_dfs_count: int = 0
        self.attempted_dfs_count: int = 0
        self.probabilistic_guesses_config_count: int = 0
        self.probabilistic_guesses_no_config_count: int = 0

        self.moves_sequence: List[Tuple[int, int, str]] = []

        # Step-by-step history for replay functionality
        # Each step is a dict with: action, cell, method, knowledge_snapshot
        self.steps_history: List[Dict[str, Any]] = []
        self._current_method: str = "first_move"  # Tracks current inference method

        # knowledge[y][x]:
        #   None     -> not revealed / not marked
        #   "M"      -> marked mine
        #   "0".."8" -> revealed number (string)
        self.knowledge: List[List[object]] = [
            [None for _ in range(game.width)] for _ in range(game.height)
        ]

        # Frontier encoding the constraint system:
        # - revealed_frontier[(x,y)] = [set_of_adjacent_unknowns, mines_remaining_around_cell]
        # - unrevealed_frontier[(ux,uy)] = set_of_adjacent_revealed_numbered_cells
        #
        # This is a bipartite view: revealed numbered cells <-> frontier unknown variables.
        self.revealed_frontier: Dict[
            Tuple[int, int], List[Union[Set[Tuple[int, int]], int]]
        ] = {}
        self.unrevealed_frontier: DefaultDict[
            Tuple[int, int], Set[Tuple[int, int]]
        ] = defaultdict(set)

        # Work queues for inference (FIFO)
        self.single_inference_queue: Deque[Tuple[int, int, str]] = deque()
        self.single_inference_set: Set[Tuple[int, int]] = set()

        self.paired_inference_queue: Deque[
            Tuple[Tuple[int, int], Tuple[int, int]]
        ] = deque()
        self.paired_inference_set: Set[
            Tuple[Tuple[int, int], Tuple[int, int]]
        ] = set()

        # Upper bound on subgroup size for SAT-style DFS inference
        self.max_dfs_subgroup_len: Union[int, float] = max_dfs_subgroup_len

    # -------------------------------------------------------------------------
    # Core functionality methods
    # -------------------------------------------------------------------------

    def neighbors(self, x: int, y: int) -> Tuple[Tuple[int, int], ...]:
        """Return precomputed neighbor coordinates for a cell."""
        return self._neighborhoods[(x, y)]

    def graph_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Build adjacency in the "numbered-cell graph" for paired inference.

        Returns other revealed numbered frontier cells that share at least one
        adjacent unknown frontier variable with (x, y).
        """
        graph_neighbors: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = {(x, y)}

        unknown_neighbors = cast(
            Set[Tuple[int, int]], self.revealed_frontier[(x, y)][0]
        )

        for unknown_cell in unknown_neighbors:
            for revealed_cell in self.unrevealed_frontier[unknown_cell]:
                if revealed_cell in seen:
                    continue
                graph_neighbors.append(revealed_cell)
                seen.add(revealed_cell)

        return graph_neighbors

    def update_frontier(self, x: int, y: int, kind: str) -> None:
        """
        Update the frontier constraint system after resolving a cell.

        Args:
            x: X-coordinate of the resolved cell.
            y: Y-coordinate of the resolved cell.
            kind: Resolution type: "M" for mine mark, "S" for newly revealed safe cell.
        """
        # (x,y) was a frontier variable: remove its edges to revealed constraints
        affected_revealed: Set[Tuple[int, int]] = self.unrevealed_frontier[(x, y)]
        del self.unrevealed_frontier[(x, y)]

        for rx, ry in affected_revealed:
            unrevealed_neighbors = cast(
                Set[Tuple[int, int]], self.revealed_frontier[(rx, ry)][0]
            )
            unmarked_mines_count = cast(int, self.revealed_frontier[(rx, ry)][1])

            unrevealed_neighbors.remove((x, y))

            if kind == "M":
                unmarked_mines_count -= 1
                self.revealed_frontier[(rx, ry)][1] = unmarked_mines_count

            if not unrevealed_neighbors:
                del self.revealed_frontier[(rx, ry)]
                continue

            self.attempted_single_count += 1
            if unmarked_mines_count == 0:
                if (rx, ry) not in self.single_inference_set:
                    self.single_inference_queue.append((rx, ry, "S"))
                    self.single_inference_set.add((rx, ry))
                else:
                    self.attempted_single_count -= 1
            elif unmarked_mines_count == len(unrevealed_neighbors):
                if (rx, ry) not in self.single_inference_set:
                    self.single_inference_queue.append((rx, ry, "M"))
                    self.single_inference_set.add((rx, ry))
                else:
                    self.attempted_single_count -= 1
            else:
                for nx, ny in self.graph_neighbors(rx, ry):
                    a: Tuple[int, int] = (rx, ry)
                    b: Tuple[int, int] = (nx, ny)
                    cand = (a, b) if a <= b else (b, a)

                    if cand not in self.paired_inference_set:
                        self.paired_inference_queue.append(cand)
                        self.paired_inference_set.add(cand)

        # If we revealed a safe numbered cell at (x,y), it becomes a new constraint
        if kind == "S":
            unrevealed_neighbors_new: Set[Tuple[int, int]] = set()
            v = self.knowledge[y][x]
            if not isinstance(v, str) or v == "M":
                raise ValueError(
                    "Expected a revealed number string at knowledge[y][x]."
                )

            unmarked_mines_count = int(v)

            for nx, ny in self.neighbors(x, y):
                if self.knowledge[ny][nx] is None:
                    unrevealed_neighbors_new.add((nx, ny))
                    self.unrevealed_frontier[(nx, ny)].add((x, y))
                elif self.knowledge[ny][nx] == "M":
                    unmarked_mines_count -= 1

            self.max_unrevealed_frontier = max(
                self.max_unrevealed_frontier, len(self.unrevealed_frontier)
            )

            if not unrevealed_neighbors_new:
                return

            self.revealed_frontier[(x, y)] = [
                unrevealed_neighbors_new,
                unmarked_mines_count,
            ]
            self.max_revealed_frontier = max(
                self.max_revealed_frontier, len(self.revealed_frontier)
            )

            self.attempted_single_count += 1
            if unmarked_mines_count == 0:
                self.single_inference_queue.append((x, y, "S"))
                self.single_inference_set.add((x, y))
                return
            elif unmarked_mines_count == len(unrevealed_neighbors_new):
                self.single_inference_queue.append((x, y, "M"))
                self.single_inference_set.add((x, y))
                return

            for nx, ny in self.graph_neighbors(x, y):
                a: Tuple[int, int] = (x, y)
                b: Tuple[int, int] = (nx, ny)
                cand = (a, b) if a <= b else (b, a)
                if cand not in self.paired_inference_set:
                    self.paired_inference_queue.append(cand)
                    self.paired_inference_set.add(cand)

    def _record_step(self, action: str, x: int, y: int) -> None:
        """Record a step for replay functionality."""
        if not self.record_steps:
            return
        import copy
        self.steps_history.append({
            "action": action,  # "reveal" or "mark"
            "cell": (x, y),
            "method": self._current_method,
            "step_number": len(self.steps_history),
            "knowledge_snapshot": copy.deepcopy(self.knowledge),
        })

    def mark_cell(self, x: int, y: int) -> Tuple[int, Dict[str, Any]]:
        """Mark a cell as a mine in solver state and propagate constraints."""
        self.knowledge[y][x] = "M"
        self.moves_sequence.append((x, y, "M"))
        self._record_step("mark", x, y)
        self.unknown_count -= 1
        self.expected_mines_count -= 1
        self.update_frontier(x, y, "M")
        return 0, {}

    def process_end_of_game_payload(
        self, status: int, game_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert a terminal engine result into the solver's metrics payload."""
        if status == -1:
            if (
                "all_mines" not in game_payload
                or "revealed_cells_count" not in game_payload
            ):
                raise ValueError(
                    "Loss payload must include 'all_mines' and 'revealed_cells_count'."
                )

            all_mines_raw = game_payload["all_mines"]
            if not isinstance(all_mines_raw, (set, frozenset, list, tuple)):
                raise ValueError("'all_mines' must be an iterable of (x,y) pairs.")

            all_mines: Set[Tuple[int, int]] = set(all_mines_raw)  # type: ignore

            return {
                "reveal_moves_count": self.reveal_moves_count,
                "moves_sequence": self.moves_sequence,
                "steps_history": self.steps_history,
                "revealed_cells_count": game_payload["revealed_cells_count"],
                "markings_count": len(all_mines) - self.expected_mines_count,
                "max_revealed_frontier": self.max_revealed_frontier,
                "max_unrevealed_frontier": self.max_unrevealed_frontier,
                "inferred_single_count": self.inferred_single_count,
                "attempted_single_count": self.attempted_single_count,
                "inferred_paired_count": self.inferred_paired_count,
                "attempted_paired_count": self.attempted_paired_count,
                "inferred_dfs_count": self.inferred_dfs_count,
                "attempted_dfs_count": self.attempted_dfs_count,
                "probabilistic_guesses_config_count": self.probabilistic_guesses_config_count,
                "probabilistic_guesses_no_config_count": self.probabilistic_guesses_no_config_count,
            }

        if status == 1:
            return {
                "reveal_moves_count": self.reveal_moves_count,
                "moves_sequence": self.moves_sequence,
                "steps_history": self.steps_history,
                "revealed_cells_count": self.board_width * self.board_height
                - self.game.mines_count,
                "markings_count": self.game.mines_count,
                "max_revealed_frontier": self.max_revealed_frontier,
                "max_unrevealed_frontier": self.max_unrevealed_frontier,
                "inferred_single_count": self.inferred_single_count,
                "attempted_single_count": self.attempted_single_count,
                "inferred_paired_count": self.inferred_paired_count,
                "attempted_paired_count": self.attempted_paired_count,
                "inferred_dfs_count": self.inferred_dfs_count,
                "attempted_dfs_count": self.attempted_dfs_count,
                "probabilistic_guesses_config_count": self.probabilistic_guesses_config_count,
                "probabilistic_guesses_no_config_count": self.probabilistic_guesses_no_config_count,
            }

        raise ValueError(f"Unexpected status for end-of-game payload: {status}")

    def reveal_cell(self, x: int, y: int) -> Tuple[int, Dict[str, Any]]:
        """
        Reveal a cell via the engine and update solver state accordingly.

        Returns:
            Tuple of (status, payload). For non-terminal reveals (status 0),
            returns the engine payload; for terminal reveals (status -1 or 1),
            returns a metrics payload.
        """
        self.moves_sequence.append((x, y, "S"))
        status, payload = self.game.reveal(x, y)
        self.reveal_moves_count += 1

        if status in (-1, 1):
            if status == -1:
                # Loss: Show complete final state with all mines
                all_mines = payload.get("all_mines", frozenset())
                for mx, my in all_mines:
                    if self.knowledge[my][mx] != "M":
                        # Unrevealed mine (not flagged) - mark as revealed mine
                        self.knowledge[my][mx] = "X"  # "X" = revealed mine (not flagged)
                # The hit mine gets special marker
                self.knowledge[y][x] = "!"  # "!" = hit mine (caused the loss)
            else:
                # Win: Copy complete final board state (all revealed cells)
                for py in range(self.board_height):
                    for px in range(self.board_width):
                        if self.game.revealed[py][px]:
                            self.knowledge[py][px] = self.game.board[py][px]
                # Flagged mines stay as "M" (correctly identified)

            # Record the final reveal step with complete state
            self._record_step("reveal", x, y)

            end_payload = self.process_end_of_game_payload(status, payload)
            return status, end_payload

        raw_cells = payload.get("revealed_cells")
        if raw_cells is None or not isinstance(raw_cells, list):
            raise ValueError("Expected payload['revealed_cells'] as a list.")

        revealed_cells = cast(List[Tuple[int, int, str]], raw_cells)

        for px, py, pv in revealed_cells:
            self.knowledge[py][px] = pv
            self.unknown_count -= 1

        # Record the step AFTER knowledge is updated so snapshot shows the result
        self._record_step("reveal", x, y)

        for px, py, _ in revealed_cells:
            self.update_frontier(px, py, "S")

        return status, payload

    # -------------------------------------------------------------------------
    # Local inference implementation
    # -------------------------------------------------------------------------

    def single_infer(
        self, x: int, y: int, infer_type: str
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Apply trivial single-constraint inference from one revealed numbered cell.

        Args:
            x: X-coordinate of the revealed numbered constraint cell.
            y: Y-coordinate of the revealed numbered constraint cell.
            infer_type: "S" to reveal all adjacent unknowns as safe,
                        "M" to mark all as mines.

        Returns:
            Tuple of (status, payload). Returns terminal status/payload if game ends,
            otherwise returns (0, {}).
        """
        if (x, y) not in self.revealed_frontier:
            self.attempted_single_count -= 1
            return 0, {}

        unrevealed_neighbors = cast(
            Set[Tuple[int, int]], self.revealed_frontier[(x, y)][0]
        )
        self.inferred_single_count += len(unrevealed_neighbors)

        if infer_type == "S":
            for nx, ny in tuple(unrevealed_neighbors):
                if self.knowledge[ny][nx] is None:
                    status, payload = self.reveal_cell(nx, ny)
                    if status in (-1, 1):
                        return status, payload
        else:  # infer_type == "M"
            for nx, ny in tuple(unrevealed_neighbors):
                self.mark_cell(nx, ny)

        return 0, {}

    def paired_infer(
        self, c1: Tuple[int, int], c2: Tuple[int, int]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Apply overlap-based inference from two revealed numbered constraints.

        Returns:
            Tuple of (status, payload) where:
            - 0 means no new information
            - 2 means new information was applied (caller should restart local inference)
            - -1/1 are terminal statuses (loss/win)
        """
        if c1 not in self.revealed_frontier or c2 not in self.revealed_frontier:
            return 0, {}

        self.attempted_paired_count += 1

        unrevealed1 = cast(Set[Tuple[int, int]], self.revealed_frontier[c1][0])
        mines1 = cast(int, self.revealed_frontier[c1][1])
        unrevealed2 = cast(Set[Tuple[int, int]], self.revealed_frontier[c2][0])
        mines2 = cast(int, self.revealed_frontier[c2][1])

        intersection = unrevealed1 & unrevealed2
        if not intersection:
            return 0, {}

        only1 = unrevealed1 - intersection
        only2 = unrevealed2 - intersection

        t_low = max(0, mines1 - len(only1), mines2 - len(only2))
        t_high = min(len(intersection), mines1, mines2)

        changed: bool = False

        if only1:
            only1_min = mines1 - t_high
            only1_max = mines1 - t_low

            if only1_max == 0:
                self.inferred_paired_count += len(only1)
                for nx, ny in only1:
                    if self.knowledge[ny][nx] is None:
                        status, payload = self.reveal_cell(nx, ny)
                        if status in (-1, 1):
                            return status, payload
                        changed = True

            elif only1_min == len(only1):
                self.inferred_paired_count += len(only1)
                for nx, ny in only1:
                    self.mark_cell(nx, ny)
                    changed = True

        if only2:
            only2_min = mines2 - t_high
            only2_max = mines2 - t_low

            if only2_max == 0:
                self.inferred_paired_count += len(only2)
                for nx, ny in only2:
                    if self.knowledge[ny][nx] is None:
                        status, payload = self.reveal_cell(nx, ny)
                        if status in (-1, 1):
                            return status, payload
                        changed = True

            elif only2_min == len(only2):
                self.inferred_paired_count += len(only2)
                for nx, ny in only2:
                    self.mark_cell(nx, ny)
                    changed = True

        return (2, {}) if changed else (0, {})

    # -------------------------------------------------------------------------
    # SAT/DFS implementation
    # -------------------------------------------------------------------------

    def _get_frontier_subgroups(
        self, max_subgroup_len: Union[int, float]
    ) -> Tuple[
        List[Tuple[List[Tuple[int, int]], FrozenSet[Tuple[int, int]]]],
        Set[Tuple[int, int]],
    ]:
        """
        Partition the revealed frontier into connected constraint components.

        Returns:
            Tuple of:
            - subgroups: (numbered_cells, variables_set) pairs, sorted by size
            - extra_variables: variables belonging to subgroups skipped due to size bound
        """
        subgroups: List[
            Tuple[List[Tuple[int, int]], FrozenSet[Tuple[int, int]]]
        ] = []
        seen_numbered: Set[Tuple[int, int]] = set()
        extra_variables: Set[Tuple[int, int]] = set()

        for start in self.revealed_frontier:
            if start in seen_numbered:
                continue

            stack: List[Tuple[int, int]] = [start]
            seen_unrevealed: Set[Tuple[int, int]] = set()
            subgroup: List[Tuple[int, int]] = []

            while stack:
                cell = stack.pop()
                seen_numbered.add(cell)
                subgroup.append(cell)

                unknowns = self.revealed_frontier[cell][0]
                unrevealed_adjacent = cast(Set[Tuple[int, int]], unknowns)

                for n in unrevealed_adjacent:
                    if n in seen_unrevealed:
                        continue
                    seen_unrevealed.add(n)

                    for nbr_num in self.unrevealed_frontier[n]:
                        if (
                            nbr_num in self.revealed_frontier
                            and nbr_num not in seen_numbered
                        ):
                            stack.append(nbr_num)
                            seen_numbered.add(nbr_num)

            if len(subgroup) > max_subgroup_len:
                extra_variables |= seen_unrevealed
                continue

            subgroups.append((subgroup, frozenset(seen_unrevealed)))

        return sorted(subgroups, key=lambda g: len(g[0])), extra_variables

    def _dfs_group(
        self,
        frontier_subgroup: List[Tuple[int, int]],
        i: int,
        assignment: Dict[Tuple[int, int], str],
        assignments_data: Dict[str, Any],
        mines_used: int,
        mines_left_total: int,
    ) -> None:
        """
        Enumerate satisfying assignments for one connected subgroup of constraints.

        This performs a bounded DFS over variable assignments that satisfy all
        numbered constraints in the subgroup.
        """
        if mines_used > mines_left_total:
            return

        if i == len(frontier_subgroup):
            mines_frequency_counts = cast(
                DefaultDict[Tuple[int, int], int],
                assignments_data["mines_frequency_counts"],
            )
            mines_config_list = cast(
                List[Tuple[Set[Tuple[int, int]], int]],
                assignments_data["mines_configurations"],
            )

            mines_configuration: Set[Tuple[int, int]] = set()
            for cell, val in assignment.items():
                if val == "M":
                    mines_configuration.add(cell)
                    mines_frequency_counts[cell] += 1

            mines_config_list.append((mines_configuration, mines_used))
            assignments_data["successful_assignments_count"] = (
                cast(int, assignments_data["successful_assignments_count"]) + 1
            )

            prev_min = cast(Union[int, float], assignments_data["min_mines_used"])
            assignments_data["min_mines_used"] = min(prev_min, mines_used)
            return

        cur_cell = frontier_subgroup[i]

        entry = self.revealed_frontier[cur_cell]
        unrevealed_adjacent = cast(Set[Tuple[int, int]], entry[0])
        expected_mines_count = cast(int, entry[1])

        assigned_mines_count = 0
        unassigned_cells: List[Tuple[int, int]] = []
        for n in unrevealed_adjacent:
            v = assignment.get(n)
            if v is None:
                unassigned_cells.append(n)
            elif v == "M":
                assigned_mines_count += 1

        needed_mines_count = expected_mines_count - assigned_mines_count
        if (
            needed_mines_count < 0
            or needed_mines_count > len(unassigned_cells)
            or mines_used + needed_mines_count > mines_left_total
        ):
            return

        for mines_tuple in itertools.combinations(unassigned_cells, needed_mines_count):
            mines_set = set(mines_tuple)
            for cell in unassigned_cells:
                assignment[cell] = "M" if cell in mines_set else "S"

            self._dfs_group(
                frontier_subgroup,
                i + 1,
                assignment,
                assignments_data,
                mines_used + needed_mines_count,
                mines_left_total,
            )

        for cell in unassigned_cells:
            del assignment[cell]

    def _identify_forced_cells(
        self,
        variables_set: AbstractSet[Tuple[int, int]],
        assignments_count: int,
        mines_frequency_counts: Dict[Tuple[int, int], int],
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Identify and apply forced variables implied by satisfying assignments.

        Returns:
            Tuple of (status, payload) where:
            - 0 indicates no forced variables found
            - 2 indicates forced moves were applied (restart local inference)
            - -1/1 indicate the game ended due to a forced safe reveal
        """
        if assignments_count == 0:
            raise RuntimeError(
                "No satisfying assignments found for this constraint set."
            )

        forced_mines: Set[Tuple[int, int]] = set()
        forced_safes: Set[Tuple[int, int]] = set()

        for v in variables_set:
            freq = mines_frequency_counts[v]
            if freq == assignments_count:
                forced_mines.add(v)
            elif freq == 0:
                forced_safes.add(v)

        if not forced_mines and not forced_safes:
            return 0, {}

        self.inferred_dfs_count += len(forced_mines) + len(forced_safes)

        for fx, fy in forced_mines:
            self.mark_cell(fx, fy)

        for fx, fy in forced_safes:
            if self.knowledge[fy][fx] is None:
                status, payload = self.reveal_cell(fx, fy)
                if status in (-1, 1):
                    return status, payload

        return 2, {}

    def _get_possible_mine_configurations(
        self, subgroups_data: List[List[Tuple[Set[Tuple[int, int]], int]]]
    ) -> Tuple[
        List[int],
        DefaultDict[Tuple[int, int], List[int]],
        DefaultDict[Tuple[int, int], int],
    ]:
        """
        Combine subgroup mine configurations into global mine configurations.

        Returns:
            Tuple of:
            - mines_per_configs: total mines used for each global configuration
            - mines_in_configs: cell -> list of config indices where cell is a mine
            - mines_frequency_counts: cell -> count of configs where cell is a mine
        """
        mines_per_configs: List[int] = []
        mines_in_configs: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        mines_frequency_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)

        cur_config_idx: int = 0

        def dfs(
            i: int, cur_config: Set[Tuple[int, int]], config_mines_count: int
        ) -> None:
            nonlocal cur_config_idx

            if config_mines_count > self.expected_mines_count:
                return

            if i == len(subgroups_data):
                mines_per_configs.append(config_mines_count)
                for cell in cur_config:
                    mines_in_configs[cell].append(cur_config_idx)
                    mines_frequency_counts[cell] += 1
                cur_config_idx += 1
                return

            for mines, subgroup_mines_count in subgroups_data[i]:
                new_mines_count = config_mines_count + subgroup_mines_count
                if new_mines_count > self.expected_mines_count:
                    continue

                added: List[Tuple[int, int]] = []
                for cell in mines:
                    cur_config.add(cell)
                    added.append(cell)

                dfs(i + 1, cur_config, new_mines_count)

                for cell in added:
                    cur_config.remove(cell)

        dfs(0, set(), 0)
        return mines_per_configs, mines_in_configs, mines_frequency_counts

    def sat_dfs_infer(
        self, max_subgroup_len: Union[int, float]
    ) -> Tuple[
        int,
        Dict[str, Any],
        List[int],
        DefaultDict[Tuple[int, int], List[int]],
        Set[Tuple[int, int]],
    ]:
        """
        Run SAT-style DFS inference over the current frontier constraint system.

        Returns:
            Tuple of (status, payload, mines_per_configs, mines_in_configs,
            unrevealed_in_configs) where:
            - status is 2 if forced moves were applied, 0 if none found, -1/1 for terminal
            - Configuration data is provided for probability guessing
        """
        self.attempted_dfs_count += 1

        mines_left_under_assignments: int = self.expected_mines_count
        subgroups_data: List[List[Tuple[Set[Tuple[int, int]], int]]] = []

        subgroups, extra_variables = self._get_frontier_subgroups(max_subgroup_len)
        if not subgroups:
            return 0, {}, [], defaultdict(list), set()

        for subgroup, seen_unrevealed in subgroups:
            assignments_data: Dict[str, Any] = {
                "mines_configurations": [],
                "mines_frequency_counts": defaultdict(int),
                "successful_assignments_count": 0,
                "min_mines_used": float("inf"),
            }

            self._dfs_group(
                subgroup, 0, {}, assignments_data, 0, mines_left_under_assignments
            )

            successful = cast(int, assignments_data["successful_assignments_count"])
            if successful == 0:
                raise RuntimeError(
                    "No satisfying assignments found for a frontier subgroup."
                )

            mines_freq = cast(
                DefaultDict[Tuple[int, int], int],
                assignments_data["mines_frequency_counts"],
            )
            status, payload = self._identify_forced_cells(
                seen_unrevealed, successful, mines_freq
            )

            if status in (-1, 1):
                return status, payload, [], defaultdict(list), set()
            if status == 2:
                return 2, {}, [], defaultdict(list), set()

            subgroup_configs = cast(
                List[Tuple[Set[Tuple[int, int]], int]],
                assignments_data["mines_configurations"],
            )
            subgroups_data.append(subgroup_configs)

            min_used = cast(Union[int, float], assignments_data["min_mines_used"])
            if not isinstance(min_used, int):
                min_used = int(min_used)
            mines_left_under_assignments -= min_used

        (
            mines_per_configs,
            mines_in_configs,
            mines_frequency_counts,
        ) = self._get_possible_mine_configurations(subgroups_data)

        if len(mines_per_configs) == 0:
            raise RuntimeError(
                "No global configurations found after combining subgroup configurations."
            )

        unrevealed_in_configs: Set[Tuple[int, int]] = (
            set(self.unrevealed_frontier) - extra_variables
        )

        status, payload = self._identify_forced_cells(
            unrevealed_in_configs, len(mines_per_configs), mines_frequency_counts
        )

        if status in (-1, 1):
            return status, payload, [], defaultdict(list), set()
        if status == 2:
            return 2, {}, [], defaultdict(list), set()

        return 0, {}, mines_per_configs, mines_in_configs, unrevealed_in_configs

    # -------------------------------------------------------------------------
    # Guessing logic implementation
    # -------------------------------------------------------------------------

    def _get_probabilities_configs(
        self,
        mines_per_configs: List[int],
        mines_in_configs: DefaultDict[Tuple[int, int], List[int]],
        unrevealed_in_configs: Set[Tuple[int, int]],
    ) -> List[Tuple[Union[Tuple[int, int], str], float]]:
        """
        Estimate mine probabilities from enumerated global configurations.

        Uses Bayesian probability estimation weighted by the number of ways
        to distribute remaining mines among floating tiles.
        """
        self.probabilistic_guesses_config_count += 1

        Y: int = self.unknown_count - len(self.unrevealed_frontier)
        M: int = self.expected_mines_count

        mines_probabilities: List[Tuple[Union[Tuple[int, int], str], float]] = []

        if not mines_per_configs:
            return mines_probabilities

        valid_mines_per_configs: List[int] = []
        for m_d in mines_per_configs:
            remaining = M - m_d
            if 0 <= remaining <= Y:
                valid_mines_per_configs.append(m_d)

        if not valid_mines_per_configs:
            return mines_probabilities

        denominator = sum(comb(Y, M - m_d) for m_d in valid_mines_per_configs)

        for cell in unrevealed_in_configs:
            numerator = 0
            for conf_idx in mines_in_configs[cell]:
                if conf_idx < 0 or conf_idx >= len(mines_per_configs):
                    continue
                m_c = mines_per_configs[conf_idx]
                remaining = M - m_c
                if 0 <= remaining <= Y:
                    numerator += comb(Y, remaining)

            mines_probabilities.append((cell, numerator / denominator))

        if Y != 0:
            numerator_float = 0.0
            for m_c in valid_mines_per_configs:
                remaining = M - m_c
                numerator_float += (remaining / Y) * comb(Y, remaining)
            mines_probabilities.append(("floating_tiles", numerator_float / denominator))

        return mines_probabilities

    def _get_probabilities_no_configs(
        self,
    ) -> List[Tuple[Union[Tuple[int, int], str], float]]:
        """
        Estimate mine probabilities without configuration enumeration (fallback).

        Approximates risk by averaging local constraint densities around each
        frontier variable.
        """
        self.probabilistic_guesses_no_config_count += 1

        densities_around_revealed_cells: Dict[Tuple[int, int], float] = {}
        for cell, entry in self.revealed_frontier.items():
            unknowns = cast(Set[Tuple[int, int]], entry[0])
            mines_remaining = cast(int, entry[1])
            density = mines_remaining / len(unknowns)
            densities_around_revealed_cells[cell] = density

        mines_probabilities: List[Tuple[Union[Tuple[int, int], str], float]] = []

        for cell, revealed_neighbors in self.unrevealed_frontier.items():
            mine_probability = sum(
                densities_around_revealed_cells[nbr] for nbr in revealed_neighbors
            ) / len(revealed_neighbors)
            mines_probabilities.append((cell, mine_probability))

        Y = self.unknown_count - len(self.unrevealed_frontier)
        if Y != 0:
            mines_probabilities.append(
                ("floating_tiles", self.expected_mines_count / self.unknown_count)
            )

        return mines_probabilities

    def guess_with_probabilities(
        self,
        mines_per_configs: List[int],
        mines_in_configs: DefaultDict[Tuple[int, int], List[int]],
        unrevealed_in_configs: Set[Tuple[int, int]],
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Choose and reveal a lowest-risk cell based on estimated mine probabilities.

        Strategy depends on self.guessing_strategy:
        - "bayesian": Use configuration-based probabilities when available, otherwise
          fall back to local-density heuristic.
        - "local_density": Always use local constraint density approximation.
        """
        if self.guessing_strategy == "local_density":
            mines_probabilities = self._get_probabilities_no_configs()
        elif mines_per_configs:
            mines_probabilities = self._get_probabilities_configs(
                mines_per_configs, mines_in_configs, unrevealed_in_configs
            )
        else:
            mines_probabilities = self._get_probabilities_no_configs()

        if not mines_probabilities:
            raise RuntimeError("No probability candidates available.")

        best_cell, _ = min(mines_probabilities, key=lambda t: t[1])

        if best_cell == "floating_tiles":
            cand_counts: Dict[Tuple[int, int], int] = defaultdict(int)

            for fx, fy in self.unrevealed_frontier:
                for cx, cy in self.neighbors(fx, fy):
                    if self.knowledge[cy][cx] is not None:
                        continue
                    if (cx, cy) in self.unrevealed_frontier:
                        continue
                    cand_counts[(cx, cy)] += 1

            if cand_counts:
                (x, y), _ = max(cand_counts.items(), key=lambda kv: kv[1])
                return self.reveal_cell(x, y)

            for y in range(self.board_height):
                for x in range(self.board_width):
                    if (
                        self.knowledge[y][x] is None
                        and (x, y) not in self.unrevealed_frontier
                    ):
                        return self.reveal_cell(x, y)

        gx, gy = cast(Tuple[int, int], best_cell)
        return self.reveal_cell(gx, gy)

    # -------------------------------------------------------------------------
    # Main solving loop
    # -------------------------------------------------------------------------

    def solve(self) -> Tuple[int, Dict[str, Any]]:
        """
        Solve the game end-to-end by iterating inference and guessing until termination.

        Returns:
            Tuple of (status, payload) where status is -1 (loss) or 1 (win),
            and payload is the solver's terminal metrics dictionary.
        """
        if self.game.mines_generation_algorithm == "safe_first_action_rule":
            first_x, first_y = 0, 0
        else:
            first_x, first_y = self.board_width // 2, self.board_height // 2

        self._current_method = "first_move"
        status, payload = self.reveal_cell(first_x, first_y)
        if status in (-1, 1):
            return status, payload

        while True:
            # 1) Apply all currently-trivial single-cell inferences
            self._current_method = "single_infer"
            while self.single_inference_queue:
                x, y, infer_type = self.single_inference_queue.popleft()
                self.single_inference_set.remove((x, y))

                status, payload = self.single_infer(x, y, infer_type)
                if status in (-1, 1):
                    return status, payload

            # 2) Try paired inferences until one produces new info
            self._current_method = "paired_infer"
            restart_single = False
            while self.paired_inference_queue:
                c1, c2 = self.paired_inference_queue.popleft()
                self.paired_inference_set.remove((c1, c2))

                status, payload = self.paired_infer(c1, c2)
                if status in (-1, 1):
                    return status, payload

                if status == 2:
                    restart_single = True
                    break

            if restart_single:
                continue

            # 3) SAT/DFS inference
            self._current_method = "dfs_infer"
            (
                status,
                payload,
                mines_per_configs,
                mines_in_configs,
                unrevealed_in_configs,
            ) = self.sat_dfs_infer(max_subgroup_len=self.max_dfs_subgroup_len)

            if status in (-1, 1):
                return status, payload
            if status == 2:
                continue

            # 4) Probabilistic guess when inference stalls
            self._current_method = "probabilistic_guess"
            status, payload = self.guess_with_probabilities(
                mines_per_configs, mines_in_configs, unrevealed_in_configs
            )
            if status in (-1, 1):
                return status, payload
