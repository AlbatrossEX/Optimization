"""How often each method leads, per evaluation, across the baseline non-smooth
cases (Running/Nonsmooth_Op.py).

For every case (one starting condition) and every function-evaluation count,
whichever of the three methods holds the lowest best-so-far wins that evaluation
(ties split by who reached the value first). This counts those wins over all
gh-0 cases of the latest Log/Logs/nonsmooth_op_*/ run and draws them as a
proportion (stacked area) plot, with a second panel for how many cases are still
running at each budget.

Saves win_count.png into the Graphs folder next to this script. Shared machinery
lives in Log/graph_common.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Log/
import graph_common as gc

NAME = "nonsmooth_op"
ENTRY = "Nonsmooth_Op"
METHODS = (0, 1, 2)  # gh 0 only: the three-way interpolation-model race

graph_dir = Path(__file__).resolve().parent / "Graphs"
graph_dir.mkdir(exist_ok=True)


def main():
    run_dir = gc.find_run_dir(NAME, entry_point=ENTRY)
    runs = [r for r in gc.load_runs(run_dir) if r["gh"] == 0]
    cases = gc.group_cases(runs, METHODS)
    if not cases:
        raise SystemExit(f"No complete three-method gh 0 cases in {run_dir}.")
    print(f"reading {run_dir.name}  ({len(cases)} complete cases)")
    gc.win_count_figure(
        cases,
        METHODS,
        str(graph_dir / "win_count.png"),
        title=f"Which method leads, per evaluation budget ({len(cases)} cases, gh 0)\n"
        "proportion of still-running cases whose best-so-far is lowest",
    )


if __name__ == "__main__":
    main()
