import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# Compares convergence speed of the three trust-region solvers
# (0 = bqmin step, 1 = best interpolation point, 2 = better of the two)
# across an open-ended set of logs in Log/Logs/. Running/main.py's suite can be
# re-run at any time with a different number of problems, starting points,
# radii, or reruns of the same case, so nothing here assumes a fixed count
# of any of those -- every grouping below is derived from what the log
# filenames actually contain.
#
# Each filename encodes: <label>_start<x0>_radius<r>_p<p>_method<m>_gh<gh>.txt
# `p` (the ratio-test exponent) and `gh_type` (the model builder) together
# identify which problem/model variant produced a log -- that pair is what
# we call the "function" below. `start` is the starting point, `radius` the
# initial trust-region radius, `method` the solver.
#
# Two kinds of figures are produced, one of each per function:
#   1. convergence_grid_<function>.png
#      A start x radius grid of small multiples; each cell overlays the
#      three methods' best-so-far curves for that exact scenario.
#   2. convergence_speed_vs_radius_<function>.png
#      Evaluations needed to reach 99% of the best objective found for a
#      scenario, aggregated (median + IQR band) over all starting points,
#      plotted against radius per method -- this isolates the radius effect.
# Comparing panels within a figure shows the radius effect; comparing the
# same figure across functions shows the function effect.

script_dir = os.path.dirname(os.path.abspath(__file__))
# the run logs live in Log/Logs, one level above this script's folder
folder = os.path.join(os.path.dirname(script_dir), "Logs")

LINE_PATTERN = re.compile(r"^(\d+),\[(.*?)\],([-+\deE.]+),\s*$")
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

METHOD_NAMES = {0: "bqmin step", 1: "interp. point", 2: "better of two"}
LINEWIDTH = 1.8
# Maximum-contrast palette (IBM colourblind-safe): black / vivid magenta /
# amber differ strongly in BOTH hue and lightness, so stacked bands and
# crossing curves stay distinguishable.
METHOD_STYLE = {
    0: dict(color="#000000", linestyle="-", linewidth=LINEWIDTH),
    1: dict(color="#DC267F", linestyle="-", linewidth=LINEWIDTH),
    2: dict(color="#FFB000", linestyle="-", linewidth=LINEWIDTH),
}
# Stacking offset of exactly 1/3 linewidth: coinciding curves overlap into
# one band whose top/bottom edges still reveal each colour.
METHOD_OFFSET_PTS = {0: LINEWIDTH / 3, 1: 0.0, 2: -LINEWIDTH / 3}
DRAW_ORDER = (0, 1, 2)

# A run counts as "converged" once its best-so-far value falls within this
# fraction of the best objective found across all methods for that scenario.
TARGET_TAU = 0.01


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
            objectives.append(float(match.group(3)))
    if not evals:
        return None, None
    # TR_function.count is a global counter shared across runs in one
    # process, so it is not reset between logs. Re-base each curve to start
    # at evaluation 1 so convergence speed is comparable across runs.
    evals = np.array(evals)
    evals = evals - evals[0] + 1
    objectives = np.minimum.accumulate(np.array(objectives))
    return evals, objectives


def evals_to_target(evals, objectives, target):
    """First evaluation count at which best-so-far <= target, or NaN if never reached."""
    reached = objectives <= target
    if not reached.any():
        return np.nan
    return float(evals[np.argmax(reached)])


def grid_figsize(nrows, ncols, cell_w=2.4, cell_h=2.0, max_w=40.0, max_h=34.0):
    """Cell-proportional figure size, capped so very large grids stay renderable."""
    return max(min(cell_w * ncols, max_w), 4.0), max(min(cell_h * nrows, max_h), 3.0)


def function_label(fn_key):
    p, gh = fn_key
    return f"p{p:g}_gh{gh}"


# --- Parse every log into (function, start, radius, method) -----------------

records = []  # dicts: function, start, radius, method, evals, objectives
for fname in sorted(os.listdir(folder)):
    m = NAME_PATTERN.match(fname)
    if not m:
        continue
    evals, objectives = read_curve(os.path.join(folder, fname))
    if evals is None:
        continue
    start = tuple(float(v) for v in FLOAT_IN_START.findall(m.group("start")))
    records.append(
        dict(
            function=(float(m.group("p")), int(m.group("gh"))),
            start=start,
            radius=float(m.group("radius")),
            method=int(m.group("method")),
            evals=evals,
            objectives=objectives,
        )
    )

if not records:
    raise SystemExit(f"No matching logs found in {folder}. Run Running/main.py first.")

functions = sorted({r["function"] for r in records})

# --- Figure 1: start x radius grid of method-overlay curves, per function ---

for fn_key in functions:
    fn_records = [r for r in records if r["function"] == fn_key]
    starts = sorted({r["start"] for r in fn_records})
    radii = sorted({r["radius"] for r in fn_records})

    # cell[(start, radius)][method] -> list of (evals, objectives), usually
    # length 1; kept as a list so reruns of the same scenario just overlay.
    cell = {}
    for r in fn_records:
        key = (r["start"], r["radius"])
        cell.setdefault(key, {}).setdefault(r["method"], []).append((r["evals"], r["objectives"]))

    all_evals = np.concatenate([r["evals"] for r in fn_records])
    all_obj = np.concatenate([r["objectives"] for r in fn_records])
    xscale, yscale = pick_scale(all_evals), pick_scale(all_obj)
    xlim, ylim = limits(all_evals, xscale), limits(all_obj, yscale)

    nrows, ncols = len(starts), len(radii)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=grid_figsize(nrows, ncols), squeeze=False, sharex=False, sharey=False
    )

    for i, start in enumerate(starts):
        for j, radius in enumerate(radii):
            ax = axes[i][j]
            runs = cell.get((start, radius), {})
            for method in DRAW_ORDER:
                stack = mtransforms.ScaledTranslation(
                    0.0, METHOD_OFFSET_PTS.get(method, 0.0) / 72.0, fig.dpi_scale_trans
                )
                for evals, objectives in runs.get(method, []):
                    ax.step(
                        evals,
                        objectives,
                        where="post",
                        transform=ax.transData + stack,
                        **METHOD_STYLE.get(method, {}),
                    )
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if not runs:
                ax.set_facecolor("#f0f0f0")
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

    handles = [
        plt.Line2D([], [], label=f"m{m}: {METHOD_NAMES[m]}", **METHOD_STYLE[m]) for m in DRAW_ORDER
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=len(DRAW_ORDER))
    fig.suptitle(
        f"Convergence grid -- function {function_label(fn_key)}\n"
        "rows = starting point, columns = radius (coinciding curves rendered as adjacent hairlines)",
        fontsize=11,
    )
    fig.supxlabel("Function evaluations", fontsize=9)
    fig.supylabel("Best objective so far", fontsize=9)
    plt.tight_layout(rect=(0.01, 0.01, 1, 0.94))
    out = os.path.join(script_dir, f"convergence_grid_{function_label(fn_key)}.png")
    plt.savefig(out, dpi=120)
    print(f"saved {out}  ({nrows}x{ncols} grid, {len(fn_records)} logs)")

# --- Figure 2: evaluations-to-target vs radius, aggregated over starts -----

fig2, axes2 = plt.subplots(1, len(functions), figsize=(6 * len(functions), 5), squeeze=False)
axes2 = axes2[0]

for ax, fn_key in zip(axes2, functions):
    fn_records = [r for r in records if r["function"] == fn_key]

    # Target accuracy per (start, radius) scenario: 99% of the way from the
    # scenario's starting objective to the best value any method reached.
    scenario_target = {}
    for (start, radius) in {(r["start"], r["radius"]) for r in fn_records}:
        scen = [r for r in fn_records if r["start"] == start and r["radius"] == radius]
        initial = scen[0]["objectives"][0]
        best_final = min(r["objectives"][-1] for r in scen)
        scenario_target[(start, radius)] = best_final + TARGET_TAU * (initial - best_final)

    radii = sorted({r["radius"] for r in fn_records})
    for method in DRAW_ORDER:
        medians, lo, hi, censored_frac = [], [], [], []
        for radius in radii:
            samples = [
                evals_to_target(r["evals"], r["objectives"], scenario_target[(r["start"], radius)])
                for r in fn_records
                if r["radius"] == radius and r["method"] == method
            ]
            valid = [s for s in samples if not np.isnan(s)]
            medians.append(np.median(valid) if valid else np.nan)
            lo.append(np.percentile(valid, 25) if valid else np.nan)
            hi.append(np.percentile(valid, 75) if valid else np.nan)
            censored_frac.append(1 - len(valid) / len(samples) if samples else np.nan)

        medians, lo, hi = np.array(medians), np.array(lo), np.array(hi)
        style = METHOD_STYLE[method]
        ax.plot(radii, medians, marker="o", label=f"m{method}: {METHOD_NAMES[method]}", **style)
        ax.fill_between(radii, lo, hi, color=style["color"], alpha=0.15, linewidth=0)
        # Flag radii where some runs never reached the target.
        for radius, med, frac in zip(radii, medians, censored_frac):
            if frac and frac > 0 and not np.isnan(med):
                ax.annotate(
                    f"{frac:.0%}\ncensored",
                    (radius, med),
                    fontsize=6,
                    color=style["color"],
                    ha="center",
                    va="bottom",
                    xytext=(0, 4),
                    textcoords="offset points",
                )

    valid_evals = np.concatenate(
        [
            [
                v
                for v in (
                    evals_to_target(r["evals"], r["objectives"], scenario_target[(r["start"], r["radius"])])
                    for r in fn_records
                )
                if not np.isnan(v)
            ]
        ]
    ) if fn_records else np.array([1.0])
    if valid_evals.size == 0:
        valid_evals = np.array([1.0])
    ax.set_xscale(pick_scale(radii))
    ax.set_yscale(pick_scale(valid_evals))
    ax.set_xlabel("Initial trust-region radius")
    ax.set_ylabel(f"Evaluations to reach {1 - TARGET_TAU:.0%} reduction\n(median, IQR band over starting points)")
    ax.set_title(f"function {function_label(fn_key)}", fontsize=10)
    ax.legend(fontsize=8)

fig2.suptitle("Convergence speed vs. initial radius, by method and function", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.94))
out2 = os.path.join(script_dir, "convergence_speed_vs_radius.png")
plt.savefig(out2, dpi=150)
print(f"saved {out2}")

plt.show()
