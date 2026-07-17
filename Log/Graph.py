"""General convergence utility: best-so-far vs evaluations for one experiment's
latest run, coloured by the GH model builder.

Usage:
  py Log/Graph.py                 # defaults to the nonsmooth_op run (has gh 0 & 1)
  py Log/Graph.py smooth_op       # any experiment name (a Log/Logs/<name>_*/ run)

Saves Log/Graphs/convergence_<name>.png. This is the thin, general-purpose
counterpart to the per-objective scripts under Log/<Objective>/; all the parsing
and drawing lives in Log/graph_common.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # Log/
import graph_common as gc

NAME = sys.argv[1] if len(sys.argv) > 1 else "nonsmooth_op"

graph_dir = Path(__file__).resolve().parent / "Graphs"
graph_dir.mkdir(exist_ok=True)


def main():
    run_dir = gc.find_run_dir(NAME)
    runs = gc.load_runs(run_dir)
    if not runs:
        raise SystemExit(f"No rankable logs in {run_dir}.")
    print(f"reading {run_dir.name}  ({len(runs)} runs)")
    gc.convergence_figure(
        runs,
        key="gh",
        styles=gc.GH_STYLE,
        names=gc.GH_NAMES,
        title=f"Convergence by GH model -- {NAME}",
        outfile=str(graph_dir / f"convergence_{NAME}.png"),
    )


if __name__ == "__main__":
    main()
