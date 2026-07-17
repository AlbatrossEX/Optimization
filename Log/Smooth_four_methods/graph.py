"""Graphs for Objective 1 — smooth function, the four trust-region methods
(Running/smooth_four_methods.py).

Reads the most recent Log/Logs/smooth_four_methods_<timestamp>/ run and saves,
into the Graphs folder next to this script:
  methods_convergence.png - best-so-far vs evaluations, coloured by method
  final_vs_radius.png     - median final objective per starting radius, per method
  winner_grid.png         - start x radius small-multiples, each case's winner stair
  winner_compare.png      - every case overlaid on one axes, coloured by leader
  win_count.png           - share of still-running cases each method leads

Graph types mirror the pre-existing Log/Non_smooth and Log/BQmin_graphing figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Log/
import graph_common as gc

NAME = "smooth_four_methods"
ENTRY = "smooth_four_methods"
METHODS = (0, 1, 2, 3)

graph_dir = Path(__file__).resolve().parent / "Graphs"
graph_dir.mkdir(exist_ok=True)


def main():
    run_dir = gc.find_run_dir(NAME, entry_point=ENTRY)
    runs = gc.load_runs(run_dir)
    if not runs:
        raise SystemExit(f"No rankable logs in {run_dir}.")
    print(f"reading {run_dir.name}  ({len(runs)} runs)")

    gc.convergence_figure(
        runs,
        key="method",
        styles=gc.METHOD_STYLE,
        names=gc.METHOD_NAMES,
        title="Smooth convergence by method (interpolation model, gh 0)",
        outfile=str(graph_dir / "methods_convergence.png"),
    )
    gc.final_vs_radius_figure(
        runs,
        str(graph_dir / "final_vs_radius.png"),
        title="Final objective vs starting radius (smooth, four methods)",
    )

    cases = gc.group_cases(runs, METHODS)
    if cases:
        gc.winner_grid_figure(
            cases,
            METHODS,
            str(graph_dir / "winner_grid.png"),
            title="Per-case winner grid (smooth, four methods)\n"
            "rows = starting point, columns = radius; each stair coloured by "
            "the method leading at that evaluation",
        )
        gc.winner_overlay_figure(
            cases,
            METHODS,
            str(graph_dir / "winner_compare.png"),
            title=f"Per-case winner over the evaluation budget "
            f"({len(cases)} cases, smooth, four methods)\n"
            "each curve is one starting condition, coloured by the method leading there",
        )
        gc.win_count_figure(
            cases,
            METHODS,
            str(graph_dir / "win_count.png"),
            title="Which method leads, per evaluation budget (smooth, four methods)\n"
            "proportion of still-running cases whose best-so-far is lowest",
        )
    else:
        print("no complete four-method cases; skipped winner figures.")


if __name__ == "__main__":
    main()
