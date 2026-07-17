"""Shared graphing helpers for the per-objective graph scripts under Log/.

The individual objective scripts (Log/Smooth_four_methods/graph.py, etc.) are
thin: they pick which run to read and which figures to draw. All the shared
machinery — locating a run's log directory, parsing logs into best-so-far
curves, the convergence / final-vs-radius / per-case-winner figures — lives here
so the four scripts stay consistent with each other and with the pre-existing
graphs (Log/Non_smooth/*, Log/BQmin_graphing/*).

Logs live in budget-keyed directories Log/Logs/<N> Evalu/<Name>/ (one directory
per experiment per evaluation budget); find_run_dir(name) returns the most
recent one (legacy Log/Logs/<name>_<timestamp>/ runs are still found as a
fallback).
"""
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # these scripts save PNGs; no interactive display needed
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]  # project root (Log/ sits under it)
sys.path.insert(0, str(ROOT))
from general_model.Optimize import latest_run_dir  # noqa: E402

# --- Names, colours, styles (IBM colourblind-safe palette) -------------------
# Same conventions as Log/Non_smooth/Nonsmooth_graph.py and
# Log/BQmin_graphing/BQ_algo_compare.py: one colour per series, solid equal-width
# lines, coincident stairs stacked by a fixed number of screen points.
METHOD_NAMES = {
    0: "bqmin step",
    1: "interp. point",
    2: "better of two",
    3: "dynamic interp.",
}
GH_NAMES = {0: "interpolation fit", 1: "random +-1 model"}

LINEWIDTH = 1.8
METHOD_STYLE = {
    0: dict(color="#000000", linestyle="-", linewidth=LINEWIDTH),
    1: dict(color="#DC267F", linestyle="-", linewidth=LINEWIDTH),
    2: dict(color="#FFB000", linestyle="-", linewidth=LINEWIDTH),
    3: dict(color="#648FFF", linestyle="-", linewidth=LINEWIDTH),
}
GH_STYLE = {
    0: dict(color="#000000", linestyle="-", linewidth=LINEWIDTH),
    1: dict(color="#648FFF", linestyle="-", linewidth=LINEWIDTH),
}
# small vertical stagger per method so exactly-coincident stairs render as
# adjacent hairlines instead of hiding each other
STACK_OFFSET_PTS = {
    0: LINEWIDTH / 2,
    1: LINEWIDTH / 6,
    2: -LINEWIDTH / 6,
    3: -LINEWIDTH / 2,
}

# Per-case winner colouring: one colour per competitor, plus a tie colour. A
# competitor holding the same best value as another is split by who REACHED it
# first; TIE only when the earliest reach is simultaneous (typically the shared
# start value). METHOD_COLOR races the four methods; GH_COLOR races the two model
# builders (gh 0 vs gh 1). The winner machinery is category-agnostic: pass
# whichever base colour/name maps match the `key` it is grouped on.
METHOD_COLOR = {0: "#000000", 1: "#DC267F", 2: "#FFB000", 3: "#648FFF"}
GH_COLOR = {0: "#000000", 1: "#648FFF"}
TIE = -1
TIE_COLOR = "#9E9E9E"
TIE_NAME = "tie (reached together)"

# The objective logs "inf" until the solver reaches a defined point; a
# numeric-only pattern would silently drop those lines and shift later
# evaluation numbers, so the value group is deliberately loose.
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
# one signed float even when negative coordinates are joined by a bare '-'
# (e.g. "0.82--1.38--2.75" -> 0.82, -1.38, -2.75)
FLOAT_IN_START = re.compile(r"-?\d+\.?\d*")


# --- Axis helpers ------------------------------------------------------------


def pick_scale(values, ratio=50.0):
    """Pick 'log' when the strictly-positive data spans more than `ratio`x, else
    'linear'. Non-finite values (inf from not-yet-finite evaluations) are ignored."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return "linear"
    positive = values[values > 0]
    if positive.size and values.min() > 0 and positive.max() / positive.min() > ratio:
        return "log"
    return "linear"


def limits(values, scale, pad=0.05):
    """Readable axis limits from the FINITE data: multiplicative padding in log,
    additive in linear. Non-finite values are drawn separately as a top bar
    (see inf_level), so they must not stretch the axis to infinity here."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:  # nothing finite to frame; a unit window keeps draws valid
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    if scale == "log":
        lo = float(values[values > 0].min())
        return lo * (1.0 - pad), hi * (1.0 + pad)
    span = hi - lo or abs(hi) or 1.0
    return lo - pad * span, hi + pad * span


def inf_level(ylo, yhi, scale):
    """Where to park non-finite (inf) values so they read as a bar along the top
    of the axis, just clear of the finite data. Returns (level, top): draw inf
    at `level` and cap the axis at `top` so the bar sits inside the frame."""
    if scale == "log":
        level = yhi * 1.12
        return level, level * 1.06
    span = (yhi - ylo) or abs(yhi) or 1.0
    level = yhi + 0.10 * span
    return level, yhi + 0.16 * span


def draw_inf_marker(ax, level, scale):
    """Mark the inf level with a faint dashed rule and an 'inf' tick label, so a
    curve pinned to the top bar is readable as 'not yet finite', not a real value."""
    ax.axhline(level, color="#9E9E9E", linestyle=":", linewidth=1.0, alpha=0.8, zorder=1)
    trans = ax.get_yaxis_transform()
    ax.text(
        0.004, level, "inf", transform=trans, fontsize=8, color="#9E9E9E",
        va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )


# --- Loading -----------------------------------------------------------------


def find_run_dir(name, entry_point=None):
    """The most recent log directory for `name`, or exit with a hint."""
    run_dir = latest_run_dir(name)
    if run_dir is None:
        hint = f"Run Running/{entry_point}.py first." if entry_point else ""
        raise SystemExit(
            f"No run directory 'Log/Logs/<N> Evalu/{name}/' found. {hint}".strip()
        )
    return run_dir


def read_curve(filepath):
    """(evaluation number, best-so-far objective) for one log, all points kept.

    Evaluations are re-based to this run's first logged evaluation (self.count is
    a shared counter that is not reset between runs). Non-finite evaluations are
    KEPT: the best-so-far is a running minimum, so they surface only as a leading
    run of inf before the first finite value, which the figures draw as a bar
    along the top of the axis (see inf_level) instead of dropping it. A run that
    is never finite cannot be ranked and returns (None, None).

    Radius-change lines (radius,<count>,<shrink|extend>,...) written by the
    solvers do not match LINE_PATTERN (no leading evaluation number followed by a
    coordinate vector), so they are skipped here.
    """
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

    evals = np.array(evals)
    evals = evals - evals[0] + 1
    objectives = np.array(objectives, dtype=float)
    if not np.isfinite(objectives).any():
        return None, None  # never finite: nothing to rank or plot
    return evals, np.minimum.accumulate(objectives)


def load_runs(run_dir):
    """Every rankable log in run_dir as a list of
    dict(start, radius, method, gh, evals, best)."""
    runs = []
    skipped = 0
    for fname in sorted(os.listdir(run_dir)):
        m = NAME_PATTERN.match(fname)  # also skips case_distribution.txt
        if not m:
            continue
        evals, best = read_curve(os.path.join(run_dir, fname))
        if evals is None:
            skipped += 1
            continue
        runs.append(
            dict(
                start=tuple(float(v) for v in FLOAT_IN_START.findall(m.group("start"))),
                radius=float(m.group("radius")),
                method=int(m.group("method")),
                gh=int(m.group("gh")),
                evals=evals,
                best=best,
            )
        )
    if skipped:
        print(f"note: {skipped} run(s) non-finite at every evaluation; excluded.")
    return runs


# --- Convergence figure (best-so-far vs evaluations) -------------------------


def convergence_figure(runs, key, styles, names, title, outfile):
    """Best-so-far vs evaluations for every run, coloured by runs[i][key]
    (key is 'method' or 'gh'). A run's leading not-yet-finite (inf) evaluations
    are drawn as a bar along the top of the axis rather than dropped."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_evals = np.concatenate([r["evals"] for r in runs])
    all_best = np.concatenate([r["best"] for r in runs])
    xscale, yscale = pick_scale(all_evals), pick_scale(all_best)
    ylo, yhi = limits(all_best, yscale)
    has_inf = not np.isfinite(all_best).all()
    level, top = inf_level(ylo, yhi, yscale) if has_inf else (yhi, yhi)

    for run in runs:
        style = styles.get(run[key], dict(color="#999999", linewidth=LINEWIDTH))
        stack = mtransforms.ScaledTranslation(
            0.0, STACK_OFFSET_PTS.get(run[key], 0.0) / 72.0, fig.dpi_scale_trans
        )
        best = np.where(np.isfinite(run["best"]), run["best"], level)
        ax.step(
            run["evals"],
            best,
            where="post",
            alpha=0.85,
            transform=ax.transData + stack,
            **style,
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*limits(all_evals, xscale))
    ax.set_ylim(ylo, top)
    if has_inf:
        draw_inf_marker(ax, level, yscale)

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best objective so far")
    ax.set_title(title)
    handles = [
        Line2D([], [], label=names[value], **styles[value])
        for value in sorted(styles)
        if any(r[key] == value for r in runs)
    ]
    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"saved {os.path.basename(outfile)}")


def final_vs_radius_figure(runs, outfile, title):
    """Median (across starting points) of the final best objective, per starting
    radius, one line per (method, gh) series."""
    series = {}  # (method, gh) -> radius -> [final best f per start]
    for run in runs:
        by_radius = series.setdefault((run["method"], run["gh"]), {})
        by_radius.setdefault(run["radius"], []).append(float(run["best"][-1]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for (method, gh), by_radius in sorted(series.items()):
        radii = np.array(sorted(by_radius))
        medians = np.array([np.median(by_radius[r]) for r in radii])
        style = dict(METHOD_STYLE[method]) if gh == 0 else dict(GH_STYLE[gh])
        if gh != 0:
            style["linestyle"] = "--"
        ax.plot(
            radii,
            medians,
            marker="o",
            markersize=4,
            label=f"method {method} ({METHOD_NAMES[method]}), {GH_NAMES[gh]}",
            **style,
        )

    ax.set_xscale("log")
    finals = [f for by_radius in series.values() for fs in by_radius.values() for f in fs]
    ax.set_yscale(pick_scale(finals))
    ax.set_xlabel("Starting trust-region radius")
    ax.set_ylabel("Median final objective (over starting points)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"saved {os.path.basename(outfile)}")


# --- Per-case winner (best-of-N, coloured by the leading method) -------------


def _best_at(run, grid):
    """Best-so-far of `run` at each evaluation in `grid` (step / forward fill).
    Past the run's last evaluation its final value is held."""
    idx = np.searchsorted(run["evals"], grid, side="right") - 1
    idx = np.clip(idx, 0, run["best"].size - 1)
    return run["best"][idx]


def group_cases(runs, categories, key="method"):
    """Group runs into per-case races over `categories` (values of run[`key`]).

    A case is one starting condition (same start and radius). The result maps a
    case id -> {category: run}, keeping only starting conditions raced by every
    category. `key` selects the competitor field: "method" races the four
    solvers, "gh" races the two model builders.

    When a category has several runs for the same starting condition — e.g. the
    random +-1 model drawn twice — every combination becomes its own race
    (cartesian over the duplicates) so no draw is dropped; those cases carry a
    draw index in their id. The common single-run-per-category case keeps the
    plain (start, radius) id that winner_grid_figure indexes on.
    """
    categories = tuple(categories)
    buckets = {}  # (start, radius) -> {category: [runs]}
    for run in runs:
        bucket = buckets.setdefault((run["start"], run["radius"]), {})
        bucket.setdefault(run[key], []).append(run)

    complete = {}
    dropped = 0
    for (start, radius), bucket in buckets.items():
        if not all(c in bucket for c in categories):
            dropped += 1
            continue
        combos = list(itertools.product(*(bucket[c] for c in categories)))
        for i, combo in enumerate(combos):
            case_runs = dict(zip(categories, combo))
            case_id = (start, radius) if len(combos) == 1 else (start, radius, i)
            complete[case_id] = case_runs
    if dropped:
        print(f"note: {dropped} case(s) not raced by all {len(categories)} "
              f"{key}s; excluded from winner figures.")
    return complete


def _first_finite_eval(run):
    """The (re-based) evaluation at which this run's best-so-far first becomes
    finite. best is a running minimum, so it is finite from here on."""
    finite = np.isfinite(run["best"])
    return int(run["evals"][int(np.argmax(finite))])


def case_winners(case_runs, categories):
    """(grid, best-of-N, winner) per evaluation for one case.

    winner[k] is the category (method or gh) leading at evaluation k; categories
    holding an equal best value are split by who reached it first, TIE only on a
    simultaneous reach. The race starts once every category has a finite
    best-so-far, so the not-yet-finite (inf) prefixes never decide a winner.
    """
    categories = tuple(categories)
    start = max(_first_finite_eval(case_runs[c]) for c in categories)
    last = max(int(case_runs[c]["evals"][-1]) for c in categories)
    if start > last:
        empty = np.array([], dtype=int)
        return empty, np.array([]), empty
    grid = np.arange(start, last + 1)
    curves = np.vstack([_best_at(case_runs[c], grid) for c in categories])
    best = curves.min(axis=0)
    # reach[i][k] = first grid index where category i attained best[k].
    # Best-so-far is non-increasing, so -curves is sorted and searchsorted
    # vectorises the lookup; a non-leader cannot have reached the min yet, so
    # argmin finds strict leaders too.
    reach = np.vstack(
        [np.searchsorted(-curves[i], -best, side="left") for i in range(len(categories))]
    )
    first = reach.min(axis=0)
    winner = np.where(
        (reach == first).sum(axis=0) == 1,
        np.array(categories)[reach.argmin(axis=0)],
        TIE,
    )
    return grid, best, winner


def _winner_segments(grid, best, winner):
    """where='post' step segments, one colour key per segment. The stair is
    carried one evaluation past the end so the final evaluation is drawn."""
    xs = np.append(grid, grid[-1] + 1)
    ys = np.append(best, best[-1])
    ws = np.append(winner, winner[-1])
    px = np.repeat(xs, 2)[1:]
    py = np.repeat(ys, 2)[:-1]
    points = np.column_stack([px, py])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    return segments, np.repeat(ws, 2)[: len(segments)]


def _colors_for(categories, base=METHOD_COLOR):
    colors = {c: base[c] for c in categories}
    colors[TIE] = TIE_COLOR
    return colors


def _categories(categories):
    return (*categories, TIE)


def _names_for(categories, base=METHOD_NAMES):
    names = {c: base[c] for c in categories}
    names[TIE] = TIE_NAME
    return names


def _grid_figsize(nrows, ncols, cell_w=2.4, cell_h=2.0, max_w=40.0, max_h=34.0):
    return max(min(cell_w * ncols, max_w), 4.0), max(min(cell_h * nrows, max_h), 3.0)


def _case_results(cases, categories):
    results = []
    for key, case_runs in cases.items():
        grid, best, winner = case_winners(case_runs, categories)
        if grid.size:
            results.append((key, grid, best, winner))
    return results


def winner_overlay_figure(
    cases, categories, outfile, title,
    colors_base=METHOD_COLOR, names_base=METHOD_NAMES, ylabel=None,
):
    """Every case overlaid on ONE axes: each drawn as the best-of-N stair,
    coloured at each evaluation by the category (method or gh) leading there, so
    exactly one colour shows at any evaluation. (The single-axes counterpart to
    winner_grid_figure's small multiples.)

    colors_base/names_base pick the palette and legend labels for the competitor
    kind: the method maps by default, GH_COLOR/GH_NAMES for a model-builder race.
    """
    results = _case_results(cases, categories)
    if not results:
        print(f"skipped {os.path.basename(outfile)}: no rankable case")
        return
    colors = _colors_for(categories, colors_base)
    legend = _categories(categories)
    names = _names_for(categories, names_base)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    segments, seg_colors, grids, bests = [], [], [], []
    for _key, grid, best, winner in results:
        segs, seg_winner = _winner_segments(grid, best, winner)
        segments.append(segs)
        seg_colors.extend(colors[w] for w in seg_winner)
        grids.append(grid)
        bests.append(best)
    ax.add_collection(
        LineCollection(np.concatenate(segments), colors=seg_colors, linewidths=1.2, alpha=0.8)
    )
    # LineCollection does not autoscale, so set the view explicitly.
    all_evals, all_best = np.concatenate(grids), np.concatenate(bests)
    xscale, yscale = pick_scale(all_evals), pick_scale(all_best)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*limits(all_evals, xscale))
    ax.set_ylim(*limits(all_best, yscale))
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(ylabel or "Best objective so far (best of the competitors)")
    ax.set_title(title)
    handles = [Line2D([], [], color=colors[c], lw=2, label=names[c]) for c in legend]
    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"saved {os.path.basename(outfile)}  ({len(results)} cases)")


def winner_grid_figure(cases, methods, outfile, title):
    """start x radius grid of small multiples, one case's winner stair per cell."""
    results = _case_results(cases, methods)
    if not results:
        print(f"skipped {os.path.basename(outfile)}: no rankable case")
        return
    colors, categories, names = _colors_for(methods), _categories(methods), _names_for(methods)

    starts = sorted({start for (start, _), *_ in results})
    radii = sorted({radius for (_, radius), *_ in results})
    by_key = {key: (grid, best, winner) for key, grid, best, winner in results}

    all_evals = np.concatenate([grid for _, grid, _, _ in results])
    all_best = np.concatenate([best for _, _, best, _ in results])
    xscale, yscale = pick_scale(all_evals), pick_scale(all_best)
    xlim, ylim = limits(all_evals, xscale), limits(all_best, yscale)

    nrows, ncols = len(starts), len(radii)
    fig, axes = plt.subplots(nrows, ncols, figsize=_grid_figsize(nrows, ncols), squeeze=False)
    for i, start in enumerate(starts):
        for j, radius in enumerate(radii):
            ax = axes[i][j]
            cell = by_key.get((start, radius))
            if cell is None:
                ax.set_facecolor("#f0f0f0")
            else:
                segments, seg_winner = _winner_segments(*cell)
                ax.add_collection(
                    LineCollection(segments, colors=[colors[w] for w in seg_winner], linewidths=1.0)
                )
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(f"radius={radius:g}", fontsize=8)
            if j == 0:
                ax.set_ylabel("start=(" + ",".join(f"{v:g}" for v in start) + ")", fontsize=6.5)
            else:
                ax.set_yticklabels([])
            if i != nrows - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)

    handles = [Line2D([], [], color=colors[c], lw=2, label=names[c]) for c in categories]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=len(categories))
    fig.suptitle(title, fontsize=11)
    fig.supxlabel("Function evaluations", fontsize=9)
    fig.supylabel("Best objective so far (best of the methods)", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.94))
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"saved {os.path.basename(outfile)}  ({nrows}x{ncols} grid, {len(results)} cases)")


def win_count_figure(cases, methods, outfile, title):
    """Share of the still-running cases each method leads, per evaluation budget."""
    results = _case_results(cases, methods)
    if not results:
        print(f"skipped {os.path.basename(outfile)}: no rankable case")
        return
    colors, categories, names = _colors_for(methods), _categories(methods), _names_for(methods)

    last = max(int(grid[-1]) for _, grid, _, _ in results)
    evals = np.arange(1, last + 1)
    counts = {c: np.zeros(last, dtype=int) for c in categories}
    active = np.zeros(last, dtype=int)
    for _key, grid, _best, winner in results:
        lo, hi = int(grid[0]) - 1, int(grid[-1])
        active[lo:hi] += 1
        for c in categories:
            counts[c][lo:hi] += winner == c

    fig, (ax, ax_n) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7), gridspec_kw=dict(height_ratios=[3, 1])
    )
    denom = np.maximum(active, 1)
    shares = [np.where(active > 0, counts[c] / denom, 0.0) for c in categories]
    ax.stackplot(evals, *shares, colors=[colors[c] for c in categories],
                 labels=[names[c] for c in categories], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of active cases led")
    ax.set_title(title)
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
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    total = sum(int(counts[c].sum()) for c in categories)
    print(f"saved {os.path.basename(outfile)}  ({total} evaluations judged)")
    for c in categories:
        won = int(counts[c].sum())
        print(f"    {names[c]:<22}: {won:8d} ({100 * won / total:5.1f}%)")
