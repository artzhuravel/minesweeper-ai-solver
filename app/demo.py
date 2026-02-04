"""
Minesweeper AI Solver - Interactive Demo

Run with: streamlit run app/demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from typing import Any, Dict, List, Tuple, Optional

from minesweeper import Minesweeper, MinesweeperSolver


def render_board_from_snapshot(
    knowledge: List[List[Any]],
    width: int,
    height: int,
    highlight_cell: Optional[Tuple[int, int]] = None,
) -> str:
    """Render a board from a knowledge snapshot for replay mode."""
    # Scale cell size based on board width
    if width >= 30:
        cell_size = 14
        font_size = "10px"
    elif width >= 25:
        cell_size = 16
        font_size = "11px"
    elif width >= 16:
        cell_size = 20
        font_size = "13px"
    else:
        cell_size = 26
        font_size = "15px"

    colors = {
        "0": "#cccccc",
        "1": "#0000ff",
        "2": "#008000",
        "3": "#ff0000",
        "4": "#000080",
        "5": "#800000",
        "6": "#008080",
        "7": "#000000",
        "8": "#808080",
        "M": "#ff0000",
        ".": "#666666",
        "F": "#ffffff",
        "X": "#ff0000",  # Revealed mine (not flagged)
        "!": "#ffffff",  # Hit mine
    }

    html = '<div style="font-family: monospace; line-height: 1.2;">'
    html += '<table style="border-collapse: collapse; margin: auto;">'

    for y in range(height):
        html += "<tr>"
        for x in range(width):
            cell_value = knowledge[y][x]

            if cell_value == "M":
                cell = "F"  # Flagged by solver
                bg = "#ffa500"
                text_color = "#ffffff"
            elif cell_value == "X":
                cell = "M"  # Revealed mine (was not flagged)
                bg = "#ffcccc"
                text_color = "#ff0000"
            elif cell_value == "!":
                cell = "M"  # Hit mine (caused loss)
                bg = "#ff0000"  # Bright red background
                text_color = "#ffffff"  # White text on red
            elif cell_value is not None:
                cell = str(cell_value)
                bg = "#f0f0f0" if cell == "0" else "#ffffff"
                text_color = colors.get(cell, "#000000")
            else:
                cell = "."
                bg = "#c0c0c0"
                text_color = "#666666"

            # Highlight current cell
            if highlight_cell and (x, y) == highlight_cell:
                border = "3px solid #ff0000"
            else:
                border = "1px solid #999"
            display = cell if cell != "0" else " "

            html += f'''<td style="
                width: {cell_size}px; height: {cell_size}px;
                text-align: center;
                background: {bg};
                border: {border};
                color: {text_color};
                font-weight: bold;
                font-size: {font_size};
            ">{display}</td>'''
        html += "</tr>"

    html += "</table></div>"
    return html


def render_board_html(
    game: Minesweeper,
    solver: Optional[MinesweeperSolver] = None,
    highlight_cell: Optional[Tuple[int, int]] = None,
    show_mines: bool = False,
) -> str:
    """Render the board as HTML with styling."""
    # Scale cell size based on board width
    if game.width >= 30:
        cell_size = 14
        font_size = "10px"
    elif game.width >= 25:
        cell_size = 16
        font_size = "11px"
    elif game.width >= 16:
        cell_size = 20
        font_size = "13px"
    else:
        cell_size = 26
        font_size = "15px"

    colors = {
        "0": "#cccccc",
        "1": "#0000ff",
        "2": "#008000",
        "3": "#ff0000",
        "4": "#000080",
        "5": "#800000",
        "6": "#008080",
        "7": "#000000",
        "8": "#808080",
        "M": "#ff0000",
        ".": "#666666",  # Dark dot for unrevealed
        "F": "#ffffff",  # White text on orange background
    }

    html = '<div style="font-family: monospace; line-height: 1.2;">'
    html += '<table style="border-collapse: collapse; margin: auto;">'

    for y in range(game.height):
        html += "<tr>"
        for x in range(game.width):
            # Determine cell display
            knowledge_val = solver.knowledge[y][x] if solver else None

            if knowledge_val == "M":
                cell = "F"  # Flagged by solver
                bg = "#ffa500"
                text_color = "#ffffff"
            elif knowledge_val == "X":
                cell = "M"  # Revealed mine (was not flagged)
                bg = "#ffcccc"
                text_color = "#ff0000"
            elif knowledge_val == "!":
                cell = "M"  # Hit mine (caused loss)
                bg = "#ff0000"
                text_color = "#ffffff"  # White text on red
            elif game.revealed[y][x]:
                cell = game.board[y][x]
                bg = "#f0f0f0" if cell == "0" else "#ffffff"
                text_color = colors.get(cell, "#000000")
            elif show_mines and game.board[y][x] == "M":
                cell = "M"
                bg = "#ffcccc"
                text_color = "#ff0000"
            else:
                cell = "."
                bg = "#c0c0c0"
                text_color = "#666666"

            # Highlight current cell
            border = "2px solid #ff0000" if (x, y) == highlight_cell else "1px solid #999"
            display = cell if cell != "0" else " "

            html += f'''<td style="
                width: {cell_size}px; height: {cell_size}px;
                text-align: center;
                background: {bg};
                border: {border};
                color: {text_color};
                font-weight: bold;
                font-size: {font_size};
            ">{display}</td>'''
        html += "</tr>"

    html += "</table></div>"
    return html


def main():
    st.set_page_config(
        page_title="Minesweeper AI Solver",
        page_icon="💣",
        layout="wide",
    )

    st.title("Minesweeper AI Solver")
    st.markdown("""
    An AI that solves Minesweeper using constraint-based inference and probabilistic guessing.
    """)

    # Sidebar configuration
    st.sidebar.header("Game Configuration")

    preset = st.sidebar.selectbox(
        "Difficulty Preset",
        ["Beginner (9x9, 10)", "Intermediate (16x16, 40)", "Expert (30x16, 99)", "Custom"],
    )

    if preset == "Beginner (9x9, 10)":
        width, height, mines = 9, 9, 10
    elif preset == "Intermediate (16x16, 40)":
        width, height, mines = 16, 16, 40
    elif preset == "Expert (30x16, 99)":
        width, height, mines = 30, 16, 99
    else:
        width = st.sidebar.slider("Width", 5, 30, 16)
        height = st.sidebar.slider("Height", 5, 30, 16)
        max_mines = width * height - 9
        mines = st.sidebar.slider("Mines", 1, max_mines, min(40, max_mines))

    algorithm = st.sidebar.selectbox(
        "Mine Generation",
        ["safe_neighborhood_rule", "safe_first_action_rule"],
        help="safe_neighborhood_rule: First click + neighbors are safe. "
             "safe_first_action_rule: Only first click is safe.",
    )

    guessing_strategy = st.sidebar.selectbox(
        "Guessing Strategy",
        ["bayesian", "local_density"],
        format_func=lambda x: "Bayesian (Recommended)" if x == "bayesian" else "Local Density",
        help="bayesian: Uses SAT/DFS configurations for precise probability estimation. "
             "local_density: Approximates probability using local constraint densities (faster but less accurate).",
    )

    max_dfs = st.sidebar.selectbox(
        "Max DFS Subgroup Size",
        [10, 20, 50, 100, "Unlimited"],
        index=4,
        help="Limit on connected constraint components for SAT/DFS inference.",
    )
    max_dfs_val: float = float("inf") if max_dfs == "Unlimited" else float(max_dfs)

    # Initialize session state
    if "game" not in st.session_state:
        st.session_state.game = None
        st.session_state.solver = None
        st.session_state.status = None
        st.session_state.step_history = []
        st.session_state.steps_history = []  # Detailed step history for replay
        st.session_state.replay_mode = False
        st.session_state.current_step = 0
        st.session_state.prev_settings = None

    # Auto-generate new game when board settings change
    current_settings = (width, height, mines, algorithm)
    if st.session_state.prev_settings != current_settings:
        st.session_state.game = Minesweeper(
            width, height, mines, mines_generation_algorithm=algorithm
        )
        st.session_state.solver = MinesweeperSolver(
            st.session_state.game, max_dfs_subgroup_len=max_dfs_val,
            guessing_strategy=guessing_strategy
        )
        st.session_state.status = None
        st.session_state.step_history = []
        st.session_state.steps_history = []
        st.session_state.replay_mode = False
        st.session_state.current_step = 0
        st.session_state.prev_settings = current_settings

    # For large boards, use vertical layout (stats below board)
    # For smaller boards, use side-by-side layout
    use_vertical_layout = width >= 30

    # Initialize containers
    col2 = None  # Will be set for side-by-side layout
    if use_vertical_layout:
        # Full width for board, stats will go below
        board_container = st.container()
    else:
        # Side-by-side layout
        if width >= 25:
            col1, col2 = st.columns([4, 1])
        elif width >= 16:
            col1, col2 = st.columns([3, 1])
        else:
            col1, col2 = st.columns([2, 1])
        board_container = col1

    with board_container:
        st.subheader("Game Board")

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("Regenerate Board", type="primary"):
                # Create a new unsolved board with the same settings
                st.session_state.game = Minesweeper(
                    width, height, mines, mines_generation_algorithm=algorithm
                )
                st.session_state.solver = MinesweeperSolver(
                    st.session_state.game, max_dfs_subgroup_len=max_dfs_val,
                    guessing_strategy=guessing_strategy
                )
                st.session_state.status = None
                st.session_state.step_history = []
                st.session_state.steps_history = []
                st.session_state.replay_mode = False
                st.session_state.current_step = 0
                st.rerun()

        with btn_col2:
            if st.button("Solve"):
                game = st.session_state.game
                if game is None:
                    # No game exists yet, create one
                    game = Minesweeper(
                        width, height, mines, mines_generation_algorithm=algorithm
                    )
                    st.session_state.game = game
                elif st.session_state.status is not None:
                    # Game was already solved, reset it to re-solve the same board
                    game.reset()

                # Create a fresh solver for the current game
                st.session_state.solver = MinesweeperSolver(
                    game, max_dfs_subgroup_len=max_dfs_val,
                    guessing_strategy=guessing_strategy
                )

                solver = st.session_state.solver
                status, payload = solver.solve()
                st.session_state.status = status
                st.session_state.payload = payload
                # Store steps history for replay
                st.session_state.steps_history = payload.get("steps_history", [])
                st.session_state.replay_mode = False
                st.session_state.current_step = len(st.session_state.steps_history) - 1
                st.rerun()

        # Replay controls (show only after solving)
        steps_history = st.session_state.steps_history
        if st.session_state.status is not None and steps_history:
            st.markdown("---")
            replay_toggle = st.checkbox(
                "Step-by-Step Replay Mode",
                value=st.session_state.replay_mode,
                key="replay_toggle"
            )
            st.session_state.replay_mode = replay_toggle

            if st.session_state.replay_mode:
                total_steps = len(steps_history)

                # Navigation controls
                nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 2])
                with nav_col1:
                    if st.button("⏮ First"):
                        st.session_state.current_step = 0
                        st.rerun()
                with nav_col2:
                    if st.button("◀ Prev"):
                        if st.session_state.current_step > 0:
                            st.session_state.current_step -= 1
                            st.rerun()
                with nav_col3:
                    if st.button("Next ▶"):
                        if st.session_state.current_step < total_steps - 1:
                            st.session_state.current_step += 1
                            st.rerun()
                with nav_col4:
                    if st.button("Last ⏭"):
                        st.session_state.current_step = total_steps - 1
                        st.rerun()

                # Step slider (1-based display)
                step_display = st.slider(
                    "Step",
                    1,
                    total_steps,
                    st.session_state.current_step + 1,
                    key="step_slider"
                )
                st.session_state.current_step = step_display - 1

                # Current step info
                current_step_data = steps_history[st.session_state.current_step]
                method_labels = {
                    "first_move": "First Move",
                    "single_infer": "Single Inference",
                    "paired_infer": "Paired Inference",
                    "dfs_infer": "DFS/SAT Inference",
                    "probabilistic_guess": "Probabilistic Guess"
                }

                action_label = "Reveal" if current_step_data["action"] == "reveal" else "Mark as Mine"
                method_label = method_labels[current_step_data["method"]]
                cell = current_step_data["cell"]

                # Check if this is the final step (game ended)
                is_final_step = st.session_state.current_step == total_steps - 1
                if is_final_step and st.session_state.status == 1:
                    st.success(f"**Step {step_display}/{total_steps}**: {action_label} cell ({cell[0]}, {cell[1]}) — *{method_label}* — **Game Won!**")
                elif is_final_step and st.session_state.status == -1:
                    st.error(f"**Step {step_display}/{total_steps}**: {action_label} cell ({cell[0]}, {cell[1]}) — *{method_label}* — **Game Lost! Hit a mine.**")
                else:
                    st.info(f"**Step {step_display}/{total_steps}**: {action_label} cell ({cell[0]}, {cell[1]}) — *{method_label}*")

        # Display board
        if st.session_state.game:
            game = st.session_state.game
            solver = st.session_state.solver

            # Render board based on mode
            if st.session_state.replay_mode and steps_history:
                current_step_data = steps_history[st.session_state.current_step]
                highlight_cell = current_step_data.get("cell")
                total_steps = len(steps_history)
                is_final_step = st.session_state.current_step == total_steps - 1

                if is_final_step and st.session_state.status is not None:
                    # Final step: show actual game state (same as normal mode) with highlight
                    html = render_board_html(game, solver, highlight_cell=highlight_cell, show_mines=True)
                else:
                    # Intermediate step: render from snapshot
                    html = render_board_from_snapshot(
                        current_step_data["knowledge_snapshot"],
                        game.width,
                        game.height,
                        highlight_cell=highlight_cell
                    )
            else:
                # Normal mode: render final state
                show_mines = st.session_state.status is not None
                html = render_board_html(game, solver, show_mines=show_mines)

            st.markdown(html, unsafe_allow_html=True)

            if not st.session_state.replay_mode:
                if st.session_state.status == 1:
                    st.success("Solved! All safe cells revealed.")
                elif st.session_state.status == -1:
                    st.error("Game Over! Hit a mine.")

            # Board legend
            st.markdown("""
            <div style="font-size: 12px; margin-top: 10px;">
            <b>Legend:</b>
            <span style="background: #c0c0c0; color: #666666; padding: 2px 6px; margin: 0 4px; font-weight: bold;">.</span> Unrevealed
            <span style="background: #f0f0f0; padding: 2px 6px; margin: 0 4px;">&nbsp;</span> Empty (0)
            <span style="color: #0000ff; font-weight: bold; margin: 0 4px;">1-8</span> Adjacent mines
            <span style="background: #ffa500; color: white; padding: 2px 6px; margin: 0 4px; font-weight: bold;">F</span> Flagged (solver marked)
            <span style="background: #ffcccc; color: #ff0000; padding: 2px 6px; margin: 0 4px; font-weight: bold;">M</span> Mine (revealed at end)
            <span style="background: #ff0000; color: white; padding: 2px 6px; margin: 0 4px; font-weight: bold;">M</span> Hit mine (caused loss)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click 'Solve' to generate and solve a new board, or 'Regenerate Board' to create an unsolved board.")

    # Stats container - either col2 (side-by-side) or new container (vertical)
    if use_vertical_layout:
        stats_container = st.container()
    else:
        assert col2 is not None
        stats_container = col2

    with stats_container:
        st.subheader("Solver Statistics")

        if st.session_state.solver and st.session_state.status is not None:
            solver = st.session_state.solver
            payload = st.session_state.payload

            # For vertical layout, use columns for metrics
            if use_vertical_layout:
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Result", "Win" if st.session_state.status == 1 else "Loss")
                with mcol2:
                    st.metric("Reveal Moves", payload.get("reveal_moves_count", "N/A"))
                with mcol3:
                    st.metric("Cells Revealed", payload.get("revealed_cells_count", "N/A"))
                with mcol4:
                    st.metric("Mines Marked", payload.get("markings_count", "N/A"))
            else:
                metrics: List[Tuple[str, Any]] = [
                    ("Result", "Win" if st.session_state.status == 1 else "Loss"),
                    ("Reveal Moves", payload.get("reveal_moves_count", "N/A")),
                    ("Cells Revealed", payload.get("revealed_cells_count", "N/A")),
                    ("Mines Marked", payload.get("markings_count", "N/A")),
                ]
                for label, value in metrics:
                    st.metric(label, value)

            st.markdown("---")
            st.markdown("**Inference Statistics**")

            inf_data = {
                "Single": (
                    payload.get("inferred_single_count", 0),
                    payload.get("attempted_single_count", 0),
                ),
                "Paired": (
                    payload.get("inferred_paired_count", 0),
                    payload.get("attempted_paired_count", 0),
                ),
                "DFS": (
                    payload.get("inferred_dfs_count", 0),
                    payload.get("attempted_dfs_count", 0),
                ),
            }

            for name, (inferred, attempted) in inf_data.items():
                rate = f"{inferred/attempted*100:.1f}%" if attempted > 0 else "N/A"
                st.text(f"{name}: {inferred} inferred / {attempted} attempted ({rate})")

            guesses = (
                payload.get("probabilistic_guesses_config_count", 0)
                + payload.get("probabilistic_guesses_no_config_count", 0)
            )
            st.text(f"Guesses: {guesses}")

        else:
            st.info("Run the solver to see statistics.")

        if not use_vertical_layout:
            st.markdown("---")
            st.subheader("Algorithm Info")
            st.markdown("""
            **Inference Strategies:**
            1. **Single**: Trivial constraint propagation (k=0 or k=|U|)
            2. **Paired**: Overlap-based deduction between constraint pairs
            3. **DFS**: SAT-style enumeration of satisfying assignments
            4. **Guess**: Bayesian probability estimation
            """)


if __name__ == "__main__":
    main()
