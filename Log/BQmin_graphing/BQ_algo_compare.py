"""Winner-coloured stair convergence for the three trust-region solvers.

Reads the run logs in Log/Logs and groups them into cases. One case is a single
scenario -- same starting point and starting radius, for one function -- raced by
all three methods (0 = bqmin step, 1 = best interpolation point, 2 = better of
the two).

Within a case the three methods are compared at every function-evaluation count
on their best-so-far objective, and the case is drawn as ONE stair (the best of
the three) coloured by whichever method leads at that evaluation, so exactly one
colour is drawn at any given number of evaluations.

Nothing assumes a fixed number of problems, starting points, radii or reruns of
a case: every grouping is derived from what the log filenames actually contain.

Each filename encodes: <label>_start<x0>_radius<r>_p<p>_method<m>_gh<gh>.txt
`p` (the ratio-test exponent) and `gh_type` (the model builder) together identify
which problem/model variant produced a log -- that pair is what we call the
"function" below.

Two figures are produced per function, into the Graphs folder next to this script:
  1. winner_grid_<function>.png
     A start x radius grid of small multiples; each cell is that one scenario's
     winner-coloured stair, so the radius effect (across columns) and the
     starting-point effect (down rows) can be read off directly.
  2. win_count_<function>.png
     The same per-evaluation verdicts counted across every case: the share of
     still-running cases each method leads, as a proportion plot.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))
# the run logs live in Log/Logs, one level above this script's folder
# Path: Log/BQmin_graphing/Con_pare_BQ&Inter.py -> go up to Log/ -> then into Logs/
folder = os.path.join(os.path.dirname(script_dir), "Logs")
graph_dir = os.path.join(script_dir, "Graphs")  # generated figures land here
os.makedirs(graph_dir, exist_ok=True)

# The value group is deliberately loose: the objective logs "inf" until the
# solver reaches a point where it is defined, and a numeric-only pattern would
# silently drop those lines -- which also shifts every later evaluation number.
LINE_PATTERN = re.compile(r"^(\d+),\[(.*?)\],([^,]*),\s*$")
NAME_PATTERN = re.compile(
    r"^(?:(?P<label>[^_]+)_)?"
    r"start(?P<start>.+?)"
    r"_radius(?P<radius>[-\d.eE]+)"
    r"_p(?P<p>[-\d.eE]+)"
    r"_method(?P<method>\d+)"
    r"_gh(?P<gh>\d+)"
    r"\.txt$"
)
# Matches one signed float even when negative coordinates are separated by a
# bare '-' instead of '-,' (e.g. "0.82--1.38--2.75" -> 0.82, -1.38, -2.75).
FLOAT_IN_START = re.compile(r"-?\d+\.?\d*")

METHODS = (0, 1, 2)
METHOD_NAMES = {0: "bqmin step", 1: "interp. point", 2: "better of two"}
# Maximum-contrast palette (IBM colourblind-safe): black / vivid magenta / amber
# differ strongly in BOTH hue and lightness, so crossing stairs stay readable.
METHOD_COLOR = {0: "#000000", 1: "#DC267F", 2: "#FFB000"}
# Methods holding the same best value are not level: the one that REACHED the
# value first was ahead the whole time, so it keeps the win. TIE remains only
# when the earliest reach is simultaneous too (typically the shared starting
# value, which every method holds from its own evaluation 1).
TIE = -1
COLORS = {**METHOD_COLOR, TIE: "#9E9E9E"}
NAMES = {**METHOD_NAMES, TIE: "tie (reached together)"}
CATEGORIES = (*METHODS, TIE)


def pick_scale(values, ratio=50.0):
    """Pick 'log' when the strictly-positive data spans more than `ratio`x, else 'linear'."""
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if positive.size and values.min() > 0 and positive.max() / positive.min() > ratio:
        return "log"
    return "linear"


def limits(values, scale, pad=0.05):
    """Readable axis limits: multiplicative padding in log, additive in linear."""
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if scale == "log":
        lo = min(values[values > 0])
        return lo * (1.0 - pad), hi * (1.0 + pad)
    span = hi - lo or abs(hi) or 1.0
    return lo - pad * span, hi + pad * span


def read_curve(filepath):
    """Return (evaluation_count, best-so-far objective) arrays for one log."""
    evals, objectives = [], []
    with open(filepath, "r") as f:
        for line in f:
            match = LINE_PATTERN.match(line)
            if not match:
                continue
            evals.append(int(match.group(1)))
            objectives.append(float(match.group(3)))  # float() parses inf/nan
    if not evals:
        return None, None

    # TR_function.count is a global counter shared across runs in one process, so
    # it is not reset between logs. Re-base on this run's FIRST logged evaluation
    # -- inf or not -- so convergence speed is comparable across runs.
    evals = np.array(evals)
    evals = evals - evals[0] + 1
    objectives = np.array(objectives, dtype=float)

    # Non-finite evaluations still cost budget (so they keep their evaluation
    # number) but carry no value to plot or rank, so they are not points on the
    # best-so-far curve. A run that is never finite cannot be ranked at all.
    finite = np.isfinite(objectives)
    if not finite.any():
        return None, None
    evals, objectives = evals[finite], objectives[finite]
    return evals, np.minimum.accumulate(objectives)


def best_at(run, grid):
    """Best-so-far of `run` at each evaluation in `grid` (step / forward fill).

    Past the run's last evaluation the run has stopped, so its final value is
    held: that is the best it ever achieves, however long the others keep going.
    """
    evals, objectives = run
    idx = np.searchsorted(evals, grid, side="right") - 1
    idx = np.clip(idx, 0, objectives.size - 1)
    return objectives[idx]


def case_winners(case_runs):
    """(grid, best-of-three, winner) per function evaluation for one case.

    winner[k] is the method leading at evaluation k. Methods holding an equal
    best value are split by who reached it first; TIE only when the earliest
    reach is simultaneous as well. The race only starts once every method has a
    finite best-so-far -- before that there is nothing to compare.
    """
    start = max(int(run[0][0]) for run in case_runs.values())
    last = max(int(run[0][-1]) for run in case_runs.values())
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
    # Segment 2k is the flat run at evaluation k and 2k+1 the drop into
    # evaluation k+1; both belong to the winner at evaluation k.
    return segments, np.repeat(ws, 2)[: len(segments)]


def function_label(fn_key):
    p, gh = fn_key
    return f"p{p:g}_gh{gh}"


def grid_figsize(nrows, ncols, cell_w=2.4, cell_h=2.0, max_w=40.0, max_h=34.0):
    """Cell-proportional figure size, capped so very large grids stay renderable."""
    return max(min(cell_w * ncols, max_w), 4.0), max(min(cell_h * nrows, max_h), 3.0)


def case_results(cases):
    """[(key, grid, best, winner)] for every case that can actually be ranked.

    Judged once here and reused by both figures, so the grid and the win counts
    can never disagree about who won an evaluation.
    """
    results = []
    for key, case_runs in cases.items():
        grid, best, winner = case_winners(case_runs)
        if grid.size:
            results.append((key, grid, best, winner))
    return results


def winner_grid_figure(fn_key, results):
    """Start x radius grid of small multiples, one case's winner stair per cell."""
    starts = sorted({start for (start, _), *_ in results})
    radii = sorted({radius for (_, radius), *_ in results})
    by_key = {key: (grid, best, winner) for key, grid, best, winner in results}

    # Cells share axis limits so they can be compared against each other directly.
    all_evals = np.concatenate([grid for _, grid, _, _ in results])
    all_best = np.concatenate([best for _, _, best, _ in results])
    xscale, yscale = pick_scale(all_evals), pick_scale(all_best)
    xlim, ylim = limits(all_evals, xscale), limits(all_best, yscale)

    nrows, ncols = len(starts), len(radii)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=grid_figsize(nrows, ncols), squeeze=False
    )

    for i, start in enumerate(starts):
        for j, radius in enumerate(radii):
            ax = axes[i][j]
            cell = by_key.get((start, radius))
            if cell is None:
                ax.set_facecolor("#f0f0f0")  # scenario never ran, or is unrankable
            else:
                segments, seg_winner = winner_segments(*cell)
                ax.add_collection(
                    LineCollection(
                        segments, colors=[COLORS[w] for w in seg_winner], linewidths=1.0
                    )
                )
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            # Only label the outer edges: keeps large grids readable.
            if i == 0:
                ax.set_title(f"radius={radius:g}", fontsize=8)
            if j == 0:
                start_txt = ",".join(f"{v:g}" for v in start)
                ax.set_ylabel(f"start=({start_txt})", fontsize=6.5)
            else:
                ax.set_yticklabels([])
            if i != nrows - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)

    handles = [Line2D([], [], color=COLORS[c], lw=2, label=NAMES[c]) for c in CATEGORIES]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=len(CATEGORIES))
    fig.suptitle(
        f"Per-case winner grid -- function {function_label(fn_key)}\n"
        "rows = starting point, columns = radius; each stair is coloured by the "
        "method leading at that evaluation",
        fontsize=11,
    )
    fig.supxlabel("Function evaluations", fontsize=9)
    fig.supylabel("Best objective so far (best of the three methods)", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.94))
    out = os.path.join(graph_dir, f"winner_grid_{function_label(fn_key)}.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}  ({nrows}x{ncols} grid, {len(results)} cases)")


def win_count_figure(fn_key, results):
    """Share of the still-running cases each method leads, per evaluation budget."""
    last = max(int(grid[-1]) for _, grid, _, _ in results)
    evals = np.arange(1, last + 1)
    counts = {c: np.zeros(last, dtype=int) for c in CATEGORIES}
    active = np.zeros(last, dtype=int)
    for _key, grid, _best, winner in results:
        # Index by evaluation number: a case's grid does not necessarily start at
        # evaluation 1 (it starts once all three methods are finite). A case only
        # votes while it is still running, so the share always describes cases
        # that are genuinely active at that budget.
        lo, hi = int(grid[0]) - 1, int(grid[-1])
        active[lo:hi] += 1
        for c in CATEGORIES:
            counts[c][lo:hi] += winner == c

    fig, (ax, ax_n) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7), gridspec_kw=dict(height_ratios=[3, 1])
    )

    denom = np.maximum(active, 1)  # active is 0 where no case has started yet
    shares = [np.where(active > 0, counts[c] / denom, 0.0) for c in CATEGORIES]
    ax.stackplot(
        evals,
        *shares,
        colors=[COLORS[c] for c in CATEGORIES],
        labels=[NAMES[c] for c in CATEGORIES],
        alpha=0.9,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of active cases led")
    ax.set_title(
        f"Which method leads, per evaluation budget -- function {function_label(fn_key)}\n"
        f"proportion of the {len(results)} cases still running whose best-so-far is lowest"
    )
    # Reversed so the legend reads in the same top-to-bottom order as the stack.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5))

    ax_n.plot(evals, active, color="#648FFF", lw=1.5)
    ax_n.set_ylim(0, active.max() * 1.1)
    ax_n.set_ylabel("Cases\nstill running")
    ax_n.set_xlabel("Function evaluations")
    ax_n.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(1, last)

    fig.tight_layout()
    out = os.path.join(graph_dir, f"win_count_{function_label(fn_key)}.png")
    fig.savefig(out, dpi=150)

    total = sum(int(counts[c].sum()) for c in CATEGORIES)
    print(f"saved {out}  ({total} evaluations judged)")
    for c in CATEGORIES:
        won = int(counts[c].sum())
        print(f"    {NAMES[c]:<24}: {won:8d} ({100 * won / total:5.1f}%)")


# --- Parse every log into (function, start, radius, method) -----------------

records = []
skipped_inf = 0
for fname in sorted(os.listdir(folder)):
    m = NAME_PATTERN.match(fname)
    if not m:
        continue
    evals, objectives = read_curve(os.path.join(folder, fname))
    if evals is None:
        skipped_inf += 1
        continue
    records.append(
        dict(
            function=(float(m.group("p")), int(m.group("gh"))),
            start=tuple(float(v) for v in FLOAT_IN_START.findall(m.group("start"))),
            radius=float(m.group("radius")),
            method=int(m.group("method")),
            run=(evals, objectives),
        )
    )

if not records:
    raise SystemExit(f"No matching logs found in {folder}. Run Running/main.py first.")
if skipped_inf:
    print(f"note: {skipped_inf} run(s) have a non-finite objective at every "
          f"evaluation and cannot be ranked.")

# --- Group into cases and draw one winner figure per function ---------------

for fn_key in sorted({r["function"] for r in records}):
    # case -> {method: run}. A rerun of the same case/method simply overwrites;
    # the race needs one curve per method.
    cases = {}
    for r in records:
        if r["function"] != fn_key:
            continue
        cases.setdefault((r["start"], r["radius"]), {})[r["method"]] = r["run"]

    complete = {k: v for k, v in cases.items() if all(m in v for m in METHODS)}
    if len(complete) < len(cases):
        print(f"note: {function_label(fn_key)}: {len(cases) - len(complete)} case(s) "
              f"not raced by all three methods.")
    if not complete:
        print(f"skipped {function_label(fn_key)}: no case has all three methods "
              f"(gh 1 builds no interpolation set, so only method 0 runs it).")
        continue

    results = case_results(complete)
    if not results:
        print(f"skipped {function_label(fn_key)}: no rankable case")
        continue
    winner_grid_figure(fn_key, results)
    win_count_figure(fn_key, results)

plt.show()
