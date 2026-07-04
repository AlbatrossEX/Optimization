import os
import re
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "Logs")
line_pattern = re.compile(r"^(\d+),\[(.*?)\],([-+\deE.]+),\s*$")

log_files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))

fig, ax = plt.subplots(figsize=(10, 6))

for log_file in log_files:
    filepath = os.path.join(folder, log_file)

    iterations = []
    objectives = []
    with open(filepath, "r") as f:
        for line in f:
            match = line_pattern.match(line)
            if not match:
                continue
            iterations.append(int(match.group(1)))
            objectives.append(float(match.group(3)))

    if not iterations:
        continue

    iterations = np.array(iterations)
    objectives = np.minimum.accumulate(np.array(objectives))

    ax.step(
        iterations,
        objectives,
        where="post",
        label=log_file,
        linewidth=1.0,
        alpha=0.6,
    )

ax.set_xscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Objective value")
ax.set_title("Convergence")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "convergence.png"), dpi=150)
plt.show()
