"""Per-case winner overlay for the baseline non-smooth suite (Running/Nonsmooth_Op.py).

Groups the interpolation-model (gh 0) runs of the latest Log/Logs/nonsmooth_op_*/
run into cases -- one case is a single starting condition (same start, radius, p)
raced by the three trust-region methods (0 = bqmin step, 1 = interpolation point,
2 = better of two). Each case is drawn as ONE stair (the best of the three)
coloured at every evaluation by the method leading there, all overlaid on one
axes.

Saves winner_compare.png into the Graphs folder next to this script. Shared
machinery lives in Log/graph_common.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Log/
import graph_common as gc

NAME = "nonsmooth_op"
ENTRY = "Nonsmooth_Op"
# gh 1 builds no interpolation set, so only method 0 runs it -- there is no
# three-way race there. This comparison is gh 0 only.
METHODS = (0, 1, 2)

graph_dir = Path(__file__).resolve().parent / "Graphs"
graph_dir.mkdir(exist_ok=True)


def main():
    run_dir = gc.find_run_dir(NAME, entry_point=ENTRY)
    runs = [r for r in gc.load_runs(run_dir) if r["gh"] == 0]
    cases = gc.group_cases(runs, METHODS)
    if not cases:
        raise SystemExit(f"No complete three-method gh 0 cases in {run_dir}.")
    print(f"reading {run_dir.name}  ({len(cases)} complete cases)")
    gc.winner_overlay_figure(
        cases,
        METHODS,
        str(graph_dir / "winner_compare.png"),
        title=f"Per-case winner over the evaluation budget ({len(cases)} cases, gh 0)\n"
        "each curve is one starting condition, coloured by the method leading there",
    )


if __name__ == "__main__":
    main()
