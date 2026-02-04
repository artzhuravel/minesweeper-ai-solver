"""Analysis and benchmarking tools for the Minesweeper solver."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .engine import Minesweeper
from .solver import MinesweeperSolver


def format_solver_knowledge(
    solver: MinesweeperSolver, *, show_coords: bool = True
) -> str:
    """
    Format the solver's current knowledge grid as a human-readable string.

    Args:
        solver: Solver instance whose knowledge will be displayed.
        show_coords: If True, include coordinate labels and a header.

    Returns:
        A text grid where unknown cells are shown as '.', and known values
        are shown as their stored symbols.
    """
    w, h = solver.board_width, solver.board_height

    def cell_char(x: int, y: int) -> str:
        v = solver.knowledge[y][x]
        if v is None:
            return "."
        if isinstance(v, str):
            return v
        return str(v)

    lines: List[str] = []
    if show_coords:
        header = " ".join(f"{x:2d}" for x in range(w))
        lines.append("   " + header)
        lines.append("   " + "-" * (3 * w - 1))

    for y in range(h):
        row = " ".join(f" {cell_char(x, y)}" for x in range(w))
        lines.append(f"{y:2d} |" + row if show_coords else row)

    return "\n".join(lines)


def run_solver_single_test(
    width: int,
    height: int,
    mines_count: int,
    mines_generation_algorithm: str,
    *,
    show_boards: bool = False,
    max_dfs_subgroup_len: float = float("inf"),
    guessing_strategy: str = "bayesian",
) -> Dict[str, object]:
    """
    Run one end-to-end game with MinesweeperSolver on a fresh Minesweeper instance.

    Args:
        width: Board width.
        height: Board height.
        mines_count: Total number of mines on the board.
        mines_generation_algorithm: Mine placement rule
            ("safe_first_action_rule" or "safe_neighborhood_rule").
        show_boards: If True, print the underlying board and the solver's final
            knowledge state.
        max_dfs_subgroup_len: Maximum allowed subgroup size for SAT/DFS inference.
        guessing_strategy: Strategy for probabilistic guessing ("bayesian" or "local_density").

    Returns:
        The solver's terminal payload augmented with "status" (-1 loss, 1 win).
    """
    game = Minesweeper(
        width, height, mines_count, mines_generation_algorithm=mines_generation_algorithm
    )
    solver = MinesweeperSolver(game, max_dfs_subgroup_len=max_dfs_subgroup_len, record_steps=False, guessing_strategy=guessing_strategy)

    status, payload = solver.solve()

    if show_boards:
        print(f"Generation mode: {mines_generation_algorithm}")
        print("Underlying board (mines visible):")
        print(game.format_board(reveal_all=True))
        print()
        print("Solver knowledge (unknowns shown as '.'):")
        print(format_solver_knowledge(solver, show_coords=True))
        print()
        print(f"Finished with status {status}.")

    if not isinstance(payload, dict):  # type: ignore[redundant-expr]
        raise TypeError(f"Expected solver payload to be dict, got {type(payload)}")

    out = dict(payload)
    out["status"] = status
    return out


def run_solver_many_tests(
    width: int,
    height: int,
    mines_count: int,
    runs: int,
    mines_generation_algorithm: str,
    *,
    max_dfs_subgroup_len: float = float("inf"),
    guessing_strategy: str = "bayesian",
) -> Dict[str, float]:
    """
    Run many independent solver games and return averaged terminal metrics plus win rate.

    Args:
        width: Board width.
        height: Board height.
        mines_count: Total number of mines on the board.
        runs: Number of independent games to run.
        mines_generation_algorithm: Mine placement rule
            ("safe_first_action_rule" or "safe_neighborhood_rule").
        max_dfs_subgroup_len: Maximum allowed subgroup size for SAT/DFS inference.
        guessing_strategy: Strategy for probabilistic guessing ("bayesian" or "local_density").

    Returns:
        Averages of solver terminal payload metrics (prefixed with "avg_"), plus:
        - win_rate
        - single_infer_per_attempt
        - paired_infer_per_attempt
        - dfs_infer_per_attempt
        - avg_guesses_total
        - avg_guesses_failed
        - guess_failure_rate
    """
    required_keys = {
        "reveal_moves_count",
        "moves_sequence",
        "revealed_cells_count",
        "markings_count",
        "max_unrevealed_frontier",
        "max_revealed_frontier",
        "inferred_single_count",
        "attempted_single_count",
        "inferred_paired_count",
        "attempted_paired_count",
        "inferred_dfs_count",
        "attempted_dfs_count",
        "probabilistic_guesses_config_count",
        "probabilistic_guesses_no_config_count",
    }

    sums: Dict[str, float] = defaultdict(float)
    wins = 0

    total_attempted_single = 0.0
    total_inferred_single = 0.0
    total_attempted_paired = 0.0
    total_inferred_paired = 0.0
    total_attempted_dfs = 0.0
    total_inferred_dfs = 0.0

    total_guesses = 0.0
    total_failed_guesses = 0.0

    for _ in range(runs):
        game = Minesweeper(
            width=width,
            height=height,
            mines_count=mines_count,
            mines_generation_algorithm=mines_generation_algorithm,
        )
        solver = MinesweeperSolver(game, max_dfs_subgroup_len=max_dfs_subgroup_len, record_steps=False, guessing_strategy=guessing_strategy)

        status, payload = solver.solve()
        if status == 1:
            wins += 1
        elif status != -1:
            raise RuntimeError(f"Unexpected solver status: {status}")

        if not isinstance(payload, dict):  # type: ignore[redundant-expr]
            raise TypeError("Expected solver payload to be a dict.")

        missing = required_keys - set(payload.keys())
        if missing:
            raise KeyError(f"Missing payload keys: {sorted(missing)}")

        sums["avg_reveal_moves_count"] += float(payload["reveal_moves_count"])
        sums["avg_revealed_cells_count"] += float(payload["revealed_cells_count"])
        sums["avg_markings_count"] += float(payload["markings_count"])
        sums["avg_max_unrevealed_frontier"] += float(
            payload["max_unrevealed_frontier"]
        )
        sums["avg_max_revealed_frontier"] += float(payload["max_revealed_frontier"])

        inf_s = float(payload["inferred_single_count"])
        att_s = float(payload["attempted_single_count"])
        inf_p = float(payload["inferred_paired_count"])
        att_p = float(payload["attempted_paired_count"])
        inf_d = float(payload["inferred_dfs_count"])
        att_d = float(payload["attempted_dfs_count"])

        sums["avg_inferred_single_count"] += inf_s
        sums["avg_attempted_single_count"] += att_s
        sums["avg_inferred_paired_count"] += inf_p
        sums["avg_attempted_paired_count"] += att_p
        sums["avg_inferred_dfs_count"] += inf_d
        sums["avg_attempted_dfs_count"] += att_d

        cfg_g = float(payload["probabilistic_guesses_config_count"])
        nocfg_g = float(payload["probabilistic_guesses_no_config_count"])
        sums["avg_probabilistic_guesses_config_count"] += cfg_g
        sums["avg_probabilistic_guesses_no_config_count"] += nocfg_g

        guesses_run = cfg_g + nocfg_g
        total_guesses += guesses_run

        if status == -1:
            total_failed_guesses += 1.0

        total_attempted_single += att_s
        total_inferred_single += inf_s
        total_attempted_paired += att_p
        total_inferred_paired += inf_p
        total_attempted_dfs += att_d
        total_inferred_dfs += inf_d

        for k, v in payload.items():
            if k in required_keys or k == "moves_sequence":
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[f"avg_{k}"] += float(v)

    out: Dict[str, float] = {k: total / runs for k, total in sums.items()}
    out["win_rate"] = wins / runs

    out["single_infer_per_attempt"] = (
        (total_inferred_single / total_attempted_single)
        if total_attempted_single > 0
        else 0.0
    )
    out["paired_infer_per_attempt"] = (
        (total_inferred_paired / total_attempted_paired)
        if total_attempted_paired > 0
        else 0.0
    )
    out["dfs_infer_per_attempt"] = (
        (total_inferred_dfs / total_attempted_dfs) if total_attempted_dfs > 0 else 0.0
    )

    out["avg_guesses_total"] = total_guesses / runs
    out["avg_guesses_failed"] = total_failed_guesses / runs
    out["guess_failure_rate"] = (
        (total_failed_guesses / total_guesses) if total_guesses > 0 else 0.0
    )

    return out


def run_solver_expert_level_analysis(
    runs: int,
    mines_generation_algorithm: str,
    *,
    max_dfs_subgroup_len: float = float("inf"),
) -> Dict[str, Dict[str, float]]:
    """
    Run aggregated solver tests on standard Minesweeper difficulty levels and plot summaries.

    Args:
        runs: Number of independent games to run per difficulty level.
        mines_generation_algorithm: Mine placement rule
            ("safe_first_action_rule" or "safe_neighborhood_rule").
        max_dfs_subgroup_len: Maximum allowed subgroup size for SAT/DFS inference.

    Returns:
        Mapping from level name to statistics dict returned by run_solver_many_tests().

    Standard difficulty levels:
        - Beginner: 9x9, 10 mines
        - Intermediate: 16x16, 40 mines
        - Expert: 30x16, 99 mines
    """
    levels: Dict[str, Tuple[int, int, int]] = {
        "beginner": (9, 9, 10),
        "intermediate": (16, 16, 40),
        "expert": (30, 16, 99),
    }

    results: Dict[str, Dict[str, float]] = {}
    for level, (w, h, m) in levels.items():
        results[level] = run_solver_many_tests(
            w, h, m, runs, mines_generation_algorithm, max_dfs_subgroup_len=max_dfs_subgroup_len
        )

    level_names = list(levels.keys())
    x = np.arange(len(level_names))

    # 1) Inferences made (by method)
    inferred_single = [results[n]["avg_inferred_single_count"] for n in level_names]
    inferred_paired = [results[n]["avg_inferred_paired_count"] for n in level_names]
    inferred_dfs = [results[n]["avg_inferred_dfs_count"] for n in level_names]

    bar_w = 0.25
    plt.figure()  # type: ignore[misc]
    plt.bar(x - bar_w, inferred_single, width=bar_w, label="single")  # type: ignore[misc]
    plt.bar(x, inferred_paired, width=bar_w, label="paired")  # type: ignore[misc]
    plt.bar(x + bar_w, inferred_dfs, width=bar_w, label="dfs")  # type: ignore[misc]
    plt.xticks(x, level_names)  # type: ignore[misc]
    plt.ylabel("Average inferred count")  # type: ignore[misc]
    plt.title("Average inferences by method (per game)")  # type: ignore[misc]
    plt.legend()  # type: ignore[misc]
    plt.tight_layout()
    plt.show()  # type: ignore[misc]

    # 2) Inference attempts (by method)
    attempted_single = [results[n]["avg_attempted_single_count"] for n in level_names]
    attempted_paired = [results[n]["avg_attempted_paired_count"] for n in level_names]
    attempted_dfs = [results[n]["avg_attempted_dfs_count"] for n in level_names]

    plt.figure()  # type: ignore[misc]
    plt.bar(x - bar_w, attempted_single, width=bar_w, label="single")  # type: ignore[misc]
    plt.bar(x, attempted_paired, width=bar_w, label="paired")  # type: ignore[misc]
    plt.bar(x + bar_w, attempted_dfs, width=bar_w, label="dfs")  # type: ignore[misc]
    plt.xticks(x, level_names)  # type: ignore[misc]
    plt.ylabel("Average attempted count")  # type: ignore[misc]
    plt.title("Average inference attempts by method (per game)")  # type: ignore[misc]
    plt.legend()  # type: ignore[misc]
    plt.tight_layout()
    plt.show()  # type: ignore[misc]

    # 3) Max frontier sizes
    max_revealed_frontier = [
        results[n]["avg_max_revealed_frontier"] for n in level_names
    ]
    max_unrevealed_frontier = [
        results[n]["avg_max_unrevealed_frontier"] for n in level_names
    ]

    plt.figure()  # type: ignore[misc]
    plt.bar(  # type: ignore[misc]
        x - bar_w / 2, max_revealed_frontier, width=bar_w, label="max_revealed_frontier"
    )
    plt.bar(  # type: ignore[misc]
        x + bar_w / 2,
        max_unrevealed_frontier,
        width=bar_w,
        label="max_unrevealed_frontier",
    )
    plt.xticks(x, level_names)  # type: ignore[misc]
    plt.ylabel("Average max frontier size")  # type: ignore[misc]
    plt.title("Average max frontier sizes (per game)")  # type: ignore[misc]
    plt.legend()  # type: ignore[misc]
    plt.tight_layout()
    plt.show()  # type: ignore[misc]

    # 4) Win rate by level
    win_rates = [results[n]["win_rate"] for n in level_names]

    plt.figure()  # type: ignore[misc]
    plt.bar(x, win_rates)  # type: ignore[misc]
    plt.xticks(x, level_names)  # type: ignore[misc]
    plt.ylabel("Win rate")  # type: ignore[misc]
    plt.ylim(0.0, 1.0)  # type: ignore[misc]
    plt.title("Win rate by difficulty level")  # type: ignore[misc]
    plt.tight_layout()
    plt.show()  # type: ignore[misc]

    return results


def summarize_inference_mix(
    results_a: Dict[str, Dict[str, float]],
    results_b: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    level: str = "expert",
) -> Dict[str, float]:
    """
    Combine (optionally) two per-level result dictionaries and compute summary statistics.

    Args:
        results_a: Dict[level_name -> metrics_dict]
        results_b: Optional second dict with the same structure (combined by simple mean).
        level: Which level to summarize (e.g., "expert").

    Returns:
        Dict with keys:
        - single_frac, paired_frac, dfs_frac, guess_frac: fractions of total operations
        - guess_success_prob: success rate of guesses
        - total_infer_plus_guesses: average total operations
        - single_forced_per_call, paired_forced_per_call, dfs_forced_per_call:
          inference productivity metrics
    """

    def get(m: Dict[str, float], k: str) -> float:
        if k not in m:
            raise KeyError(f"Missing key {k!r} in metrics for level {level!r}.")
        return float(m[k])

    if level not in results_a:
        raise KeyError(f"Level {level!r} not found in results_a.")
    m_a = results_a[level]

    if results_b is None:
        m = m_a
    else:
        if level not in results_b:
            raise KeyError(f"Level {level!r} not found in results_b.")
        m_b = results_b[level]
        m = {
            k: (float(m_a.get(k, 0.0)) + float(m_b.get(k, 0.0))) / 2.0
            for k in set(m_a) | set(m_b)
        }

    s = get(m, "avg_inferred_single_count")
    p = get(m, "avg_inferred_paired_count")
    d = get(m, "avg_inferred_dfs_count")
    g = get(m, "avg_guesses_total")

    total = s + p + d + g
    if total == 0.0:
        raise ZeroDivisionError(
            "Total (single+paired+dfs+guesses) is 0; cannot compute fractions."
        )

    if g > 0.0 and "avg_guesses_failed" in m:
        guess_success_prob = 1.0 - (float(m["avg_guesses_failed"]) / g)
    else:
        guess_success_prob = 1.0 - get(m, "guess_failure_rate")

    att_s = get(m, "avg_attempted_single_count")
    att_p = get(m, "avg_attempted_paired_count")
    att_d = get(m, "avg_attempted_dfs_count")

    if att_s == 0.0:
        raise ZeroDivisionError("avg_attempted_single_count is 0.")
    if att_p == 0.0:
        raise ZeroDivisionError("avg_attempted_paired_count is 0.")
    if att_d == 0.0:
        raise ZeroDivisionError("avg_attempted_dfs_count is 0.")

    return {
        "single_frac": s / total,
        "paired_frac": p / total,
        "dfs_frac": d / total,
        "guess_frac": g / total,
        "guess_success_prob": guess_success_prob,
        "total_infer_plus_guesses": total,
        "single_forced_per_call": s / att_s,
        "paired_forced_per_call": p / att_p,
        "dfs_forced_per_call": d / att_d,
    }
