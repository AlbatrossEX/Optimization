import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "Logs")
line_pattern = re.compile(r"^(\d+),\[(.*?)\],([-+\deE.]+),\s*$")
gh_pattern = re.compile(r"gh(\d+)")

# Skip the live/orphan log (New.txt) the same way cleanup_logs does.
log_files = sorted(
    f for f in os.listdir(folder) if f.endswith(".txt") and f != "New.txt"
)

# Curves are discriminated ONLY by gh_type (the model builder): every gh 0 run
# is one colour and every gh 1 run the other, regardless of method.
GH_NAMES = {0: "gh 0 (interpolation model)", 1: "gh 1 (random +-1 model)"}
LINEWIDTH = 1.8
# Two maximum-contrast, colourblind-safe colours (IBM palette): blue vs orange
# differ strongly in both hue and lightness, so the two gh types are unmistakable.
GH_STYLE = {
    0: dict(color="#648FFF", linestyle="-", linewidth=LINEWIDTH),
    1: dict(color="#FE6100", linestyle="-", linewidth=LINEWIDTH),
}
# Small vertical stagger per gh_type (1/2 linewidth) so that coinciding curves
# of different gh types render as adjacent bands instead of hiding each other.
GH_OFFSET_PTS = {0: LINEWIDTH / 2, 1: -LINEWIDTH / 2}


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

    g = gh_pattern.search(log_file)
    gh = int(g.group(1)) if g else None
    curves.append((gh, log_file, iterations, objectives))

if not curves:
    raise SystemExit(f"No non-empty logs found in {folder}.")

fig, ax = plt.subplots(figsize=(10, 6))

# fallback colours for logs without a parseable gh_type
fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, (gh, log_file, iterations, objectives) in enumerate(curves):
    if gh in GH_STYLE:
        style = dict(GH_STYLE[gh])
    else:
        style = dict(color=fallback[i % len(fallback)], linewidth=1.5)
    stack = mtransforms.ScaledTranslation(
        0.0, GH_OFFSET_PTS.get(gh, 0.0) / 72.0, fig.dpi_scale_trans
    )
    ax.step(
        iterations,
        objectives,
        where="post",
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
ax.set_title("Convergence by gh_type (curves coloured only by model builder)")
# One legend entry per gh_type present, rather than one per log file.
present_gh = [g for g in GH_STYLE if any(c[0] == g for c in curves)]
handles = [Line2D([0], [0], label=GH_NAMES[g], **GH_STYLE[g]) for g in present_gh]
if handles:
    ax.legend(handles=handles, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "convergence.png"), dpi=150)
plt.show()
