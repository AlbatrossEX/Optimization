"""Is the fitted G the gradient at x, or the coefficient at the origin?

Check against calfun's analytic gradient, then rerun the Bard comparison
with the re-centered gradient G + H @ x.

See bqmin_analysis.md in this folder for the findings.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from construct_functions import build_smooth_problem
from function.calfun import calfun
from Smooth.models.bqmin import bqmin

rng = np.random.default_rng(0)

print("--- gradient frame check (nprob=2, f exactly quadratic; then Bard) ---")
for nprob in (2, 8):
    prob = build_smooth_problem(np.zeros(3), m=15, nprob=nprob)
    for delta in (0.01, 0.1, 1.0):
        errs_raw, errs_centered = [], []
        for _ in range(20):
            x = rng.uniform(-2, 2, 3)
            g, h = prob.GH(x, delta)
            _, _, grad_true = calfun(x, 15, nprob, probtype="smooth", num_outs=3)
            errs_raw.append(np.linalg.norm(g - grad_true) / (np.linalg.norm(grad_true) + 1e-12))
            errs_centered.append(np.linalg.norm(g + h @ x - grad_true) / (np.linalg.norm(grad_true) + 1e-12))
        print(f"nprob={nprob} delta={delta:<5}: rel err  raw G: {np.median(errs_raw):9.3g}   "
              f"G+Hx: {np.median(errs_centered):9.3g}")

print("\n--- Bard comparison with re-centered gradient ---")
prob = build_smooth_problem(np.zeros(3), m=15, nprob=8)
RADIUS_RANGE = (0.01, 5.0)
N_TRIALS = 800
N_BINS = 10
rng = np.random.default_rng(42)
radii = np.exp(rng.uniform(np.log(RADIUS_RANGE[0]), np.log(RADIUS_RANGE[1]), N_TRIALS))
adv_raw = np.empty(N_TRIALS)
adv_fix = np.empty(N_TRIALS)
for t in range(N_TRIALS):
    x = rng.uniform(-2, 2, 3)
    r = radii[t]
    g, h = prob.GH(x, r)
    f_interp = float(np.min(prob.f_poised))
    bound = r * np.ones(3)
    s_raw, _ = bqmin(h, g, -bound, bound)
    s_fix, _ = bqmin(h, g + h @ x, -bound, bound)
    adv_raw[t] = f_interp - float(prob.f(x + np.asarray(s_raw, dtype=float)))
    adv_fix[t] = f_interp - float(prob.f(x + np.asarray(s_fix, dtype=float)))

edges = np.geomspace(*RADIUS_RANGE, N_BINS + 1)
idx = np.clip(np.digitize(radii, edges) - 1, 0, N_BINS - 1)
print("radius bin        raw: median adv / win%     recentered: median adv / win%")
for b in range(N_BINS):
    m = idx == b
    print(f"[{edges[b]:7.3g},{edges[b+1]:7.3g})   "
          f"{np.median(adv_raw[m]):10.4g} / {100*np.mean(adv_raw[m] > 0):5.1f}    "
          f"{np.median(adv_fix[m]):10.4g} / {100*np.mean(adv_fix[m] > 0):5.1f}")
