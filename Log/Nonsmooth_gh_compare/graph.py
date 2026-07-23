"""Graphs for Objective 2 — non-smooth function, method 0 with GH 0 vs GH 1
(Running/nonsmooth_gh_compare.py).

Reads the most recent Log/Logs/nonsmooth_gh_compare_<timestamp>/ run and saves,
into the Graphs folder next to this script:
  gh_convergence.png  - best-so-far vs evaluations, coloured by GH model builder
                        (gh 0 = interpolation fit, gh 1 = random +-1 model)
  final_vs_radius.png - median final objective per starting radius, one line per
                        GH model builder
  winner_compare.png  - every case overlaid on one axes, coloured by the GH model
                        leading there (each gh 1 draw raced against gh 0)

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
GH_TYPES = (0, 1)  # the two model builders raced in winner_compare.png

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

    # Per-case winner overlay, raced by GH model builder (gh 0 vs gh 1). gh 1 is
    # drawn twice, so each of its draws races the shared gh 0 run as its own case.
    cases = gc.group_cases(runs, GH_TYPES, key="gh")
    if cases:
        gc.winner_overlay_figure(
            cases,
            GH_TYPES,
            str(graph_dir / "winner_compare.png"),
            title=f"Per-case winner over the evaluation budget "
            f"({len(cases)} cases, non-smooth, GH 0 vs GH 1)\n"
            "each curve is one starting condition, coloured by the GH model leading there",
            colors_base=gc.GH_COLOR,
            names_base=gc.GH_NAMES,
            ylabel="Best objective so far (best of the two GH models)",
        )
    else:
        print("no complete gh 0 vs gh 1 cases; skipped winner_compare.png.")
    
    gc.win_count_figure(
            cases,
            GH_TYPES,
            str(graph_dir / "win_count.png"),
            title="Which method leads, per evaluation budget\n"
            "proportion of still-running cases whose best-so-far is lowest",
        )


if __name__ == "__main__":
    main()
