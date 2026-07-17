"""Winner-coloured convergence for the baseline smooth suite's three solvers
(Running/Smooth_Op.py).

Reads the latest Log/Logs/smooth_op_*/ run (archived legacy runs live in
Log/Logs/smooth_op_legacy_*/) and groups its logs into cases -- one case is a
single scenario (same start and radius) raced by the three trust-region methods
(0 = bqmin step, 1 = best interpolation point, 2 = better of the two).

Within a case the three methods are compared at every function-evaluation count
on their best-so-far objective. Two figures land in the Graphs folder next to
this script:
  winner_grid_smooth.png  - a start x radius grid of small multiples, each cell
                            that scenario's winner-coloured stair
  win_count_smooth.png    - the same verdicts counted across all cases: the share
                            of still-running cases each method leads

Shared machinery lives in Log/graph_common.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Log/
import graph_common as gc

NAME = "smooth_op"
ENTRY = "Smooth_Op"
METHODS = (0, 1, 2)

graph_dir = Path(__file__).resolve().parent / "Graphs"
graph_dir.mkdir(exist_ok=True)


def main():
    run_dir = gc.find_run_dir(NAME, entry_point=ENTRY)
    runs = gc.load_runs(run_dir)
    cases = gc.group_cases(runs, METHODS)
    if not cases:
        raise SystemExit(f"No complete three-method cases in {run_dir}.")
    print(f"reading {run_dir.name}  ({len(cases)} complete cases)")

    gc.winner_grid_figure(
        cases,
        METHODS,
        str(graph_dir / "winner_grid_smooth.png"),
        title="Per-case winner grid (smooth baseline, methods 0/1/2)\n"
        "rows = starting point, columns = radius; each stair coloured by the "
        "method leading at that evaluation",
    )
    gc.win_count_figure(
        cases,
        METHODS,
        str(graph_dir / "win_count_smooth.png"),
        title="Which method leads, per evaluation budget (smooth baseline)\n"
        "proportion of still-running cases whose best-so-far is lowest",
    )


if __name__ == "__main__":
    main()
