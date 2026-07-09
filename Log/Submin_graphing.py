import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# Compares convergence speed of the three trust-region solvers
# (0 = bqmin step, 1 = best interpolation point, 2 = better of the two)
# across scenarios that differ in both starting point and radius. Each log is
# one (scenario, method) run produced by main.py; we group by scenario so every
# subplot overlays the three methods on the same problem.

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "Logs")

# e.g. "A_start1-1-1_radius0.1_p1_method0.txt"
name_pattern = re.compile(
    r"^(?P<label>[^_]+)_start(?P<start>.+?)_radius(?P<radius>[\d.]+)_p[\d.]+_method(?P<method>\d+)\.txt$"
)
line_pattern = re.compile(r"^(\d+),\[(.*?)\],([-+\deE.]+),\s*$")

METHOD_NAMES = {
    0: "bqmin step",
    1: "interp. point",
    2: "better of two",
}
# All three methods are solid lines of equal width. Coinciding curves stay
# visible by stacking: each method is shifted vertically by a fixed number of
# screen points (constant on screen, independent of the axis scale), so exact
# overlaps render as three parallel bands instead of hiding one another.
LINEWIDTH = 1.8
# Maximum-contrast palette (IBM colourblind-safe): black / vivid magenta /
# amber differ strongly in BOTH hue and lightness (dark -> medium -> light),
# so stacked bands and crossing curves are unmistakable.
METHOD_STYLE = {
    0: dict(color="#000000", linestyle="-", linewidth=LINEWIDTH),
    1: dict(color="#DC267F", linestyle="-", linewidth=LINEWIDTH),
    2: dict(color="#FFB000", linestyle="-", linewidth=LINEWIDTH),
}
# Stacking offset of exactly 1/3 linewidth: coinciding curves overlap into one
# band whose top/bottom edges still reveal each colour, with no visible shift.
METHOD_OFFSET_PTS = {0: LINEWIDTH / 3, 1: 0.0, 2: -LINEWIDTH / 3}
DRAW_ORDER = (0, 1, 2)


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
            match = line_pattern.match(line)
            if not match:
                continue
            evals.append(int(match.group(1)))
            objectives.append(float(match.group(3)))
    if not evals:
        return None, None
    # TR_function.count is a global counter on the shared Function_object, so it
    # is not reset between the runs in one main.py process. Re-base each curve to
    # start at evaluation 1 so convergence speed is comparable across runs.
    evals = np.array(evals)
    evals = evals - evals[0] + 1
    objectives = np.minimum.accumulate(np.array(objectives))
    return evals, objectives


# Collect logs grouped by scenario, preserving discovery order.
scenarios = {}  # label -> {"start":, "radius":, "runs": {method: (evals, obj)}}
for fname in sorted(os.listdir(folder)):
    m = name_pattern.match(fname)
    if not m:
        continue
    evals, obj = read_curve(os.path.join(folder, fname))
    if evals is None:
        continue
    label = m.group("label")
    scn = scenarios.setdefault(
        label,
        {"start": m.group("start"), "radius": m.group("radius"), "runs": {}},
    )
    scn["runs"][int(m.group("method"))] = (evals, obj)

if not scenarios:
    raise SystemExit(f"No matching logs found in {folder}. Run main.py first.")

labels = sorted(scenarios)
fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
axes = axes[0]

# Shared, data-driven scales/limits so the panels stay visually comparable.
all_evals = np.concatenate(
    [e for scn in scenarios.values() for e, _ in scn["runs"].values()]
)
all_obj = np.concatenate(
    [o for scn in scenarios.values() for _, o in scn["runs"].values()]
)
xscale = pick_scale(all_evals)
yscale = pick_scale(all_obj)
xlim = limits(all_evals, xscale)
ylim = limits(all_obj, yscale)

for ax, label in zip(axes, labels):
    scn = scenarios[label]
    for method in DRAW_ORDER:
        if method not in scn["runs"]:
            continue
        evals, obj = scn["runs"][method]
        stack = mtransforms.ScaledTranslation(
            0.0, METHOD_OFFSET_PTS.get(method, 0.0) / 72.0, fig.dpi_scale_trans
        )
        ax.step(
            evals,
            obj,
            where="post",
            label=f"m{method}: {METHOD_NAMES.get(method, method)}",
            transform=ax.transData + stack,
            **METHOD_STYLE.get(method, {}),
        )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best objective so far")
    # Order the legend by method number regardless of draw order.
    handles, lbls = ax.get_legend_handles_labels()
    order = np.argsort(lbls)
    ax.legend(
        [handles[i] for i in order],
        [lbls[i] for i in order],
        fontsize=8,
        loc="upper right",
    )
    ax.set_title(
        f"Scenario {label}\nstart=[{scn['start']}], radius={scn['radius']}",
        fontsize=10,
    )

fig.suptitle(
    "Convergence speed by method, across starting points and radii\n"
    "(coinciding curves rendered as adjacent hairlines)",
    fontsize=12,
)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(os.path.join(script_dir, "Submin_convergence.png"), dpi=150)
plt.show()
