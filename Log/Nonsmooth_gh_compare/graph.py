"""Graphs for Objective 2 — non-smooth function, method 0 with GH 0 vs GH 1
(Running/nonsmooth_gh_compare.py).

Reads the most recent Log/Logs/nonsmooth_gh_compare_<timestamp>/ run and saves,
into the Graphs folder next to this script:
  gh_convergence.png  - best-so-far vs evaluations, coloured by GH model builder
                        (gh 0 = interpolation fit, gh 1 = random +-1 model)
  final_vs_radius.png - median final objective per starting radius, one line per
                        GH model builder

Both series are the same trust-region solver (method 0, bqmin step): the only
thing that differs is the model builder, so any gap between the curves is the
model builder and nothing else. Graph types mirror the pre-existing
Log/Non_smooth figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Log/
import graph_common as gc

NAME = "nonsmooth_gh_compare"
ENTRY = "nonsmooth_gh_compare"

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
        key="gh",
        styles=gc.GH_STYLE,
        names=gc.GH_NAMES,
        title="Non-smooth convergence by GH model (method 0, bqmin step)",
        outfile=str(graph_dir / "gh_convergence.png"),
    )
    gc.final_vs_radius_figure(
        runs,
        str(graph_dir / "final_vs_radius.png"),
        title="Final objective vs starting radius (non-smooth, GH 0 vs GH 1)",
    )


if __name__ == "__main__":
    main()
