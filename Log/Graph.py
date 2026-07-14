import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "Logs")
line_pattern = re.compile(r"^(\d+),\[(.*?)\],([-+\deE.]+),\s*$")
method_pattern = re.compile(r"method(\d+)")

# Skip the live/orphan log (New.txt) the same way cleanup_logs does.
log_files = sorted(
    f for f in os.listdir(folder) if f.endswith(".txt") and f != "New.txt"
)

METHOD_NAMES = {0: "bqmin step", 1: "interp. point", 2: "better of two"}
# One colour per method, all solid and equal width. Coinciding curves stay
# visible by stacking: each method is shifted vertically by a fixed number of
# screen points, so exact overlaps render as parallel bands instead of hiding.
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


curves = []  # (method, label, evals, objectives)
for log_file in log_files:
    filepath = os.path.join(folder, log_file)

    iterations, objectives = [], []
    with open(filepath, "r") as f:
        for line in f:
            match = line_pattern.match(line)
            if not match:
                continue
            iterations.append(int(match.group(1)))
            objectives.append(float(match.group(3)))

    if not iterations:
        continue

    # Re-base to evaluation 1 (TR_function.count is a shared global counter that
    # is not reset between runs) and take the best-so-far envelope.
    iterations = np.array(iterations)
    iterations = iterations - iterations[0] + 1
    objectives = np.minimum.accumulate(np.array(objectives))

    m = method_pattern.search(log_file)
    method = int(m.group(1)) if m else None
    curves.append((method, log_file, iterations, objectives))

if not curves:
    raise SystemExit(f"No non-empty logs found in {folder}.")

fig, ax = plt.subplots(figsize=(10, 6))

# fallback colours for logs without a parseable method
fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, (method, log_file, iterations, objectives) in enumerate(curves):
    if method in METHOD_STYLE:
        style = dict(METHOD_STYLE[method])
    else:
        style = dict(color=fallback[i % len(fallback)], linewidth=1.5)
    stack = mtransforms.ScaledTranslation(
        0.0, METHOD_OFFSET_PTS.get(method, 0.0) / 72.0, fig.dpi_scale_trans
    )
    ax.step(
        iterations,
        objectives,
        where="post",
        # label=log_file,
        alpha=0.85,
        transform=ax.transData + stack,
        **style,
    )

all_evals = np.concatenate([c[2] for c in curves])
all_obj = np.concatenate([c[3] for c in curves])
xscale = pick_scale(all_evals)
yscale = pick_scale(all_obj)
ax.set_xscale(xscale)
ax.set_yscale(yscale)
ax.set_xlim(*limits(all_evals, xscale))
ax.set_ylim(*limits(all_obj, yscale))

ax.set_xlabel("Function evaluations")
ax.set_ylabel("Best objective so far")
ax.set_title("Convergence (coinciding curves rendered as adjacent hairlines)")
ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "convergence.png"), dpi=150)
plt.show()
