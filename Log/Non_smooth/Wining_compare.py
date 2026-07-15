"""Per-case winner curves for the non-smooth suite (interpolation model, gh 0).

Groups the N-prefixed logs from Log/Logs into cases -- one case is a single
starting condition (same starting point, starting radius and p), solved by all
three trust-region methods (0 = bqmin step, 1 = interpolation point,
2 = better of two).

For each case the three methods are compared at every function-evaluation count
on their best-so-far objective. The case is drawn as ONE curve (the best of the
three) whose colour at each evaluation is the method that is ahead there, so
exactly one colour is drawn at any given number of evaluations.

Saves winner_compare.png into the Graphs folder next to this script.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from Nonsmooth_graph import pick_scale, limits

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(os.path.dirname(script_dir), "Logs")  # Log/Logs
# every figure this folder's scripts generate lands here (Win_count.py imports it)
graph_dir = os.path.join(script_dir, "Graphs")
os.makedirs(graph_dir, exist_ok=True)

# The value group is deliberately loose: the objective logs "inf" until the
# solver reaches a point where it is defined, and a numeric-only pattern would
# silently drop those lines -- which also shifts every later evaluation number.
line_pattern = re.compile(r"^(\d+),\[(.*?)\],([^,]*),\s*$")
# gh 1 builds no interpolation set, so only method 0 runs it -- there is no
# three-way race to judge there. This comparison is gh 0 only.
name_pattern = re.compile(
    r"^N\d+_start(.+?)_radius([\d.eE+-]+)_p([\d.eE+-]+)_method(\d+)_gh0\.txt$"
)

METHODS = (0, 1, 2)
METHOD_NAMES = {0: "bqmin step", 1: "interp. point", 2: "better of two"}
# Same IBM colourblind-safe palette as Nonsmooth_graph.py / Graph.py.
METHOD_COLOR = {0: "#000000", 1: "#DC267F", 2: "#FFB000"}
# Methods holding the same best value are not level: the one that REACHED the
# value first was ahead the whole time, so it keeps the win. TIE remains only
# when the earliest reach is simultaneous too (typically the shared starting
# value, which every method holds from its own evaluation 1).
TIE = -1
COLORS = {**METHOD_COLOR, TIE: "#9E9E9E"}
NAMES = {**METHOD_NAMES, TIE: "tie (reached together)"}
CATEGORIES = (*METHODS, TIE)


def load_gh0_runs():
    """Parse every gh 0 non-smooth log into (case, method, evals, best-so-far)."""
    runs, all_inf = [], []
    for log_file in sorted(os.listdir(folder)):
        name = name_pattern.match(log_file)
        if not name:
            continue
        case = (name.group(1), float(name.group(2)), float(name.group(3)))
        method = int(name.group(4))

        evals, objectives = [], []
        with open(os.path.join(folder, log_file), "r") as f:
            for line in f:
                match = line_pattern.match(line)
                if not match:
                    continue
                evals.append(int(match.group(1)))
                objectives.append(float(match.group(3)))  # float() parses inf/nan
        if not evals:
            continue

        # TR_function.count is a shared counter across the whole suite, so re-base
        # on this run's first logged evaluation -- inf or not -- to get a truthful
        # "evaluations into this run" axis.
        evals = np.array(evals)
        evals = evals - evals[0] + 1
        objectives = np.array(objectives, dtype=float)

        # Non-finite evaluations still cost budget (so they keep their evaluation
        # number) but carry no value to plot or rank, so they are not points on
        # the best-so-far curve. A run that is never finite cannot be ranked.
        finite = np.isfinite(objectives)
        if not finite.any():
            all_inf.append(log_file)
            continue
        evals, objectives = evals[finite], objectives[finite]
        best = np.minimum.accumulate(objectives)
        runs.append(dict(case=case, method=method, evals=evals, best=best))

    if all_inf:
        print(
            f"note: {len(all_inf)} gh 0 run(s) have a non-finite objective at every "
            f"evaluation and cannot be ranked; their cases are excluded."
        )
    return runs


def group_cases(runs):
    """case -> {method: run}, keeping only cases raced by all three methods."""
    cases = {}
    for run in runs:
        cases.setdefault(run["case"], {})[run["method"]] = run
    complete = {k: v for k, v in cases.items() if all(m in v for m in METHODS)}
    if len(complete) < len(cases):
        print(
            f"note: {len(cases) - len(complete)} case(s) dropped - not raced by all "
            f"three methods."
        )
    return complete


def best_at(run, grid):
    """Best-so-far of `run` at each evaluation in `grid` (step / forward fill).

    Past the run's last evaluation the run has stopped, so its final value is
    held: that is the best it ever achieves, however long the others keep going.
    """
    idx = np.searchsorted(run["evals"], grid, side="right") - 1
    idx = np.clip(idx, 0, run["best"].size - 1)
    return run["best"][idx]


def case_winners(case_runs):
    """(grid, best-of-three, winner) per function evaluation for one case.

    winner[k] is the method leading at evaluation k. Methods holding an equal
    best value are split by who reached it first; TIE only when the earliest
    reach is simultaneous as well.

    The race only starts once every method has a finite best-so-far: before that
    there is nothing to compare, so the grid begins at the latest of the three
    first-finite evaluations rather than at evaluation 1.
    """
    start = max(int(run["evals"][0]) for run in case_runs.values())
    last = max(int(run["evals"][-1]) for run in case_runs.values())
    if start > last:
        empty = np.array([], dtype=int)
        return empty, np.array([]), empty
    grid = np.arange(start, last + 1)
    curves = np.vstack([best_at(case_runs[m], grid) for m in METHODS])
    best = curves.min(axis=0)
    # reach[m][k] = first grid index where method m's best-so-far attained
    # best[k]. Best-so-far is non-increasing, so -curves is sorted and
    # searchsorted vectorises the lookup. A method not holding the minimum at k
    # cannot have reached it yet (reach > k), so the argmin finds the strict
    # leader too -- no separate strict/tie cases needed.
    reach = np.vstack(
        [np.searchsorted(-curves[i], -best, side="left") for i in range(len(METHODS))]
    )
    first = reach.min(axis=0)
    winner = np.where(
        (reach == first).sum(axis=0) == 1,
        np.array(METHODS)[reach.argmin(axis=0)],
        TIE,
    )
    return grid, best, winner


def winner_segments(grid, best, winner):
    """Line segments of the where='post' step curve, one colour key per segment.

    The stair is carried one evaluation past the end: with where='post' a value
    is drawn as the flat run leading to the NEXT evaluation, so the final
    evaluation would otherwise never be drawn at all.
    """
    xs = np.append(grid, grid[-1] + 1)
    ys = np.append(best, best[-1])
    ws = np.append(winner, winner[-1])

    px = np.repeat(xs, 2)[1:]
    py = np.repeat(ys, 2)[:-1]
    points = np.column_stack([px, py])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    # Segment 2k is the horizontal run at evaluation k and 2k+1 the step down to
    # evaluation k+1; both belong to the winner at evaluation k.
    seg_winner = np.repeat(ws, 2)[: len(segments)]
    return segments, seg_winner


def winner_figure(cases, outfile="winner_compare.png"):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    segments, colors, grids, bests = [], [], [], []
    for case_runs in cases.values():
        grid, best, winner = case_winners(case_runs)
        segs, seg_winner = winner_segments(grid, best, winner)
        if not len(segs):
            continue
        segments.append(segs)
        colors.extend(COLORS[w] for w in seg_winner)
        grids.append(grid)
        bests.append(best)

    ax.add_collection(
        LineCollection(np.concatenate(segments), colors=colors, linewidths=1.2, alpha=0.8)
    )

    # LineCollection does not autoscale the axes, so set the view explicitly.
    all_evals = np.concatenate(grids)
    all_best = np.concatenate(bests)
    xscale, yscale = pick_scale(all_evals), pick_scale(all_best)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*limits(all_evals, xscale))
    ax.set_ylim(*limits(all_best, yscale))

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best objective so far (best of the three methods)")
    ax.set_title(
        f"Per-case winner over the evaluation budget ({len(cases)} cases, gh 0)\n"
        "each curve is one starting condition, coloured by the method leading there"
    )
    handles = [Line2D([], [], color=COLORS[c], lw=2, label=NAMES[c]) for c in CATEGORIES]
    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, outfile), dpi=150)
    print(f"saved {outfile}")


if __name__ == "__main__":
    cases = group_cases(load_gh0_runs())
    if not cases:
        raise SystemExit(f"No complete three-method gh 0 cases found in {folder}.")
    winner_figure(cases)
    plt.show()
