"""
Quickstart example for the Minesweeper AI Solver.

This script demonstrates basic usage of the solver.
"""

from minesweeper import (
    Minesweeper,
    MinesweeperSolver,
    run_solver_many_tests,
)


def main():
    print("=" * 60)
    print("Minesweeper AI Solver - Quickstart Example")
    print("=" * 60)

    # Example 1: Solve a single game
    print("\n1. Solving a single Intermediate game (16x16, 40 mines)...")
    print("-" * 60)

    game = Minesweeper(
        width=16,
        height=16,
        mines_count=40,
        mines_generation_algorithm="safe_neighborhood_rule",
    )

    solver = MinesweeperSolver(game)
    status, payload = solver.solve()

    result = "WON" if status == 1 else "LOST"
    print(f"Result: {result}")
    print(f"Reveal moves: {payload['reveal_moves_count']}")
    print(f"Cells revealed: {payload['revealed_cells_count']}")
    print(f"Mines marked: {payload['markings_count']}")
    print(f"Single inferences: {payload['inferred_single_count']}")
    print(f"Paired inferences: {payload['inferred_paired_count']}")
    print(f"DFS inferences: {payload['inferred_dfs_count']}")

    guesses = (
        payload["probabilistic_guesses_config_count"]
        + payload["probabilistic_guesses_no_config_count"]
    )
    print(f"Probabilistic guesses: {guesses}")

    # Example 2: Show final board state
    print("\n2. Final board state:")
    print("-" * 60)
    print(game.format_board(reveal_all=True))

    # Example 3: Run multiple games for statistics
    print("\n3. Running 50 games for win rate statistics...")
    print("-" * 60)

    results = run_solver_many_tests(
        width=16,
        height=16,
        mines_count=40,
        runs=50,
        mines_generation_algorithm="safe_neighborhood_rule",
    )

    print(f"Win rate: {results['win_rate']*100:.1f}%")
    print(f"Average moves per game: {results['avg_reveal_moves_count']:.1f}")
    print(f"Average guesses per game: {results['avg_guesses_total']:.1f}")
    print(f"Guess failure rate: {results['guess_failure_rate']*100:.1f}%")

    # Example 4: Compare difficulty levels
    print("\n4. Win rates by difficulty level (10 games each)...")
    print("-" * 60)

    difficulties = [
        ("Beginner", 9, 9, 10),
        ("Intermediate", 16, 16, 40),
        ("Expert", 30, 16, 99),
    ]

    for name, w, h, m in difficulties:
        results = run_solver_many_tests(
            width=w,
            height=h,
            mines_count=m,
            runs=10,
            mines_generation_algorithm="safe_neighborhood_rule",
        )
        print(f"{name:15s} ({w}x{h}, {m:2d} mines): {results['win_rate']*100:5.1f}% win rate")

    print("\n" + "=" * 60)
    print("Done! See README.md for more detailed usage instructions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
