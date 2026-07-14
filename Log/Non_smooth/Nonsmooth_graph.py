"""Convergence graphs for the non-smooth suite (Running/Nonsmooth_Op.py).

Reads the N-prefixed logs from Log/Logs and saves three figures next to this
script:
  methods_convergence.png  - interpolation-model (gh 0) runs, coloured by
                             trust-region method
  gh_convergence.png       - method 0 runs, interpolation fit vs random +-1
                             model
  final_vs_radius.png      - median final objective per starting radius, one
                             line per (method, gh) series
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(os.path.dirname(script_dir), "Logs")  # Log/Logs

line_pattern = re.compile(r"^(\d+),\[(.*?)\],([-+\deE.]+),\s*$")
name_pattern = re.compile(
    r"^N\d+_start.+_radius([\d.eE+-]+)_p[\d.eE+-]+_method(\d+)_gh(\d+)\.txt$"
)

METHOD_NAMES = {0: "bqmin step", 1: "interp. point", 2: "better of two"}
GH_NAMES = {0: "interpolation fit", 1: "random +-1 model"}
# Same convergence-plot conventions as Log/Graph.py: one colour per series,
# solid equal-width lines, and coinciding curves stacked by a fixed number of
# screen points so exact overlaps render as adjacent hairlines instead of
# hiding each other. Colours are IBM colourblind-safe.
LINEWIDTH = 1.8
METHOD_STYLE = {
    0: dict(color="#000000", linestyle="-", linewidth=LINEWIDTH),
    1: dict(color="#DC267F", linestyle="-", linewidth=LINEWIDTH),
    2: dict(color="#FFB000", linestyle="-", linewidth=LINEWIDTH),
}
GH_STYLE = {
    0: dict(color="#000000", linestyle="-", linewidth=LINEWIDTH),
    1: dict(color="#648FFF", linestyle="-", linewidth=LINEWIDTH),
}
STACK_OFFSET_PTS = {0: LINEWIDTH / 3, 1: 0.0, 2: -LINEWIDTH / 3}


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


def load_runs():
    """Parse every non-smooth (N-prefixed) log into
    (radius, method, gh, evals, best-so-far objective) records."""
    runs = []
    for log_file in sorted(os.listdir(folder)):
        name = name_pattern.match(log_file)
        if not name:
            continue
        radius, method, gh = float(name.group(1)), int(name.group(2)), int(name.group(3))

        evals, objectives = [], []
        with open(os.path.join(folder, log_file), "r") as f:
            for line in f:
                match = line_pattern.match(line)
                if not match:
                    continue
                evals.append(int(match.group(1)))
                objectives.append(float(match.group(3)))
        if not evals:
            continue

        # Re-base to evaluation 1 (TR_function.count is a shared counter that is
        # not reset between runs) and take the best-so-far envelope.
        evals = np.array(evals)
        evals = evals - evals[0] + 1
        best = np.minimum.accumulate(np.array(objectives))
        runs.append(dict(radius=radius, method=method, gh=gh, evals=evals, best=best))
    return runs


def convergence_figure(runs, key, styles, names, title, outfile):
    """Best-so-far vs evaluations for every run, coloured by runs[i][key]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for run in runs:
        stack = mtransforms.ScaledTranslation(
            0.0, STACK_OFFSET_PTS.get(run[key], 0.0) / 72.0, fig.dpi_scale_trans
        )
        ax.step(
            run["evals"],
            run["best"],
            where="post",
            alpha=0.85,
            transform=ax.transData + stack,
            **styles[run[key]],
        )

    all_evals = np.concatenate([r["evals"] for r in runs])
    all_best = np.concatenate([r["best"] for r in runs])
    xscale, yscale = pick_scale(all_evals), pick_scale(all_best)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*limits(all_evals, xscale))
    ax.set_ylim(*limits(all_best, yscale))

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
    fig.savefig(os.path.join(script_dir, outfile), dpi=150)
    print(f"saved {outfile}")


def final_vs_radius_figure(runs, outfile):
    """Median (across starting points) of the final best objective, per
    starting radius, one line per (method, gh) series."""
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
    yscale = pick_scale(finals)
    ax.set_yscale(yscale)

    ax.set_xlabel("Starting trust-region radius")
    ax.set_ylabel("Median final objective (over starting points)")
    ax.set_title("Final objective vs starting radius (non-smooth suite)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(script_dir, outfile), dpi=150)
    print(f"saved {outfile}")


if __name__ == "__main__":
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No non-smooth (N-prefixed) logs found in {folder}.")

    # 1) Trust-region method comparison on the interpolation model.
    convergence_figure(
        [r for r in runs if r["gh"] == 0],
        key="method",
        styles=METHOD_STYLE,
        names=METHOD_NAMES,
        title="Non-smooth convergence by method (interpolation model, gh 0)",
        outfile="methods_convergence.png",
    )
    # 2) GH model comparison: both builders drive the same method-0 solver.
    convergence_figure(
        [r for r in runs if r["method"] == 0],
        key="gh",
        styles=GH_STYLE,
        names=GH_NAMES,
        title="Non-smooth convergence by GH model (method 0, bqmin step)",
        outfile="gh_convergence.png",
    )
    # 3) How the starting radius affects each (method, gh) series.
    final_vs_radius_figure(runs, outfile="final_vs_radius.png")
    plt.show()
