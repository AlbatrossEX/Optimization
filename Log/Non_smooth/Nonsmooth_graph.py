"""Convergence graphs for the baseline non-smooth suite (Running/Nonsmooth_Op.py).

Reads the latest Log/Logs/nonsmooth_op_*/ run (the archived legacy runs live in
Log/Logs/nonsmooth_op_legacy_*/) and saves into the Graphs folder next to this
script:
  methods_convergence.png - interpolation-model (gh 0) runs, coloured by method
  gh_convergence.png      - method 0 runs, interpolation fit vs random +-1 model
  final_vs_radius.png     - median final objective per starting radius, per series

All shared machinery (log parsing, styles, figure builders) lives in
Log/graph_common.py; this script only picks the run and the figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Log/
import graph_common as gc

NAME = "nonsmooth_op"
ENTRY = "Nonsmooth_Op"

graph_dir = Path(__file__).resolve().parent / "Graphs"
graph_dir.mkdir(exist_ok=True)


def main():
    run_dir = gc.find_run_dir(NAME, entry_point=ENTRY)
    runs = gc.load_runs(run_dir)
    if not runs:
        raise SystemExit(f"No rankable logs in {run_dir}.")
    print(f"reading {run_dir.name}  ({len(runs)} runs)")

    # 1) trust-region method comparison on the interpolation model (gh 0)
    gc.convergence_figure(
        [r for r in runs if r["gh"] == 0],
        key="method",
        styles=gc.METHOD_STYLE,
        names=gc.METHOD_NAMES,
        title="Non-smooth convergence by method (interpolation model, gh 0)",
        outfile=str(graph_dir / "methods_convergence.png"),
    )
    # 2) GH model comparison: both builders drive the same method-0 solver
    gc.convergence_figure(
        [r for r in runs if r["method"] == 0],
        key="gh",
        styles=gc.GH_STYLE,
        names=gc.GH_NAMES,
        title="Non-smooth convergence by GH model (method 0, bqmin step)",
        outfile=str(graph_dir / "gh_convergence.png"),
    )
    # 3) how the starting radius affects each (method, gh) series
    gc.final_vs_radius_figure(
        runs,
        str(graph_dir / "final_vs_radius.png"),
        title="Final objective vs starting radius (non-smooth baseline suite)",
    )


if __name__ == "__main__":
    main()
