"""
Minesweeper AI Solver

A constraint-based Minesweeper solver using multiple inference strategies:
- Single inference: Trivial constraint propagation
- Paired inference: Overlap-based deduction between constraint pairs
- SAT/DFS inference: Exhaustive enumeration of satisfying assignments
- Probabilistic guessing: Bayesian estimation when no forced moves exist
"""

from .engine import Minesweeper, play_cli
from .solver import MinesweeperSolver
from .analysis import (
    format_solver_knowledge,
    run_solver_single_test,
    run_solver_many_tests,
    run_solver_expert_level_analysis,
    summarize_inference_mix,
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "Minesweeper",
    "MinesweeperSolver",
    # CLI
    "play_cli",
    # Analysis functions
    "format_solver_knowledge",
    "run_solver_single_test",
    "run_solver_many_tests",
    "run_solver_expert_level_analysis",
    "summarize_inference_mix",
]
