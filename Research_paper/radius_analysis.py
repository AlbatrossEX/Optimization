"""Why does bqmin's advantage collapse near radius ~1 on the Bard problem?

Three experiments, same harness as Bqmin_compare.py:
  A. control: nprob=2 (linear rank-1 residuals -> f exactly quadratic).
     If the crossover is generic model error, it must VANISH here.
  B. nprob=12 (Box 3D: smooth exponentials, no poles near the sample box).
     Crossover should exist but be milder / elsewhere.
  C. nprob=8 (Bard). Diagnostic: does the trust region touch a pole plane
     x1*tmp2 + x2*tmp3 = 0, and does that predict bqmin's loss?

See bqmin_analysis.md in this folder for the findings.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from construct_functions import build_smooth_problem
from Smooth.models.bqmin import bqmin

DIM = 3
RADIUS_RANGE = (0.01, 5.0)
X_RANGE = (-2.0, 2.0)
N_TRIALS = 800
N_BINS = 10

# Bard denominator coefficients (tmp2, tmp3) for i = 0..14
BARD_COEF = []
for i in range(15):
    tmp2 = 15 - i
    tmp3 = min(i + 1, 15 - i)
    BARD_COEF.append((tmp2, tmp3))
BARD_COEF = np.array(sorted(set(BARD_COEF)), dtype=float)  # 8 distinct planes


def pole_in_box(x, radius):
    """True if some Bard pole plane crosses the inf-norm box of given radius at x."""
    d = BARD_COEF[:, 0] * x[1] + BARD_COEF[:, 1] * x[2]
    reach = radius * (BARD_COEF[:, 0] + BARD_COEF[:, 1])
    return bool(np.any(np.abs(d) <= reach))


def run(problem, label, pole_diag=False):
    rng = np.random.default_rng(42)
    radii = np.exp(rng.uniform(np.log(RADIUS_RANGE[0]),
                               np.log(RADIUS_RANGE[1]), N_TRIALS))
    adv = np.empty(N_TRIALS)        # f_interp - f_bqmin
    pred_ok = np.empty(N_TRIALS)    # actual reduction vs model-predicted reduction
    pole_hit = np.zeros(N_TRIALS, dtype=bool)
    contaminated = np.zeros(N_TRIALS, dtype=bool)  # poised set spans >=1e3x value range

    for t in range(N_TRIALS):
        x = rng.uniform(*X_RANGE, DIM)
        r = radii[t]
        fx = float(problem.f(x))
        g, h = problem.GH(x, r)
        f_interp = float(np.min(problem.f_poised))
        bound = r * np.ones(DIM)
        step, model_val = bqmin(h, g, -bound, bound)
        step = np.asarray(step, dtype=float)
        f_bq = float(problem.f(x + step))
        adv[t] = f_interp - f_bq
        # model predicted f(x+step) ~ fx + model_val; compare with reality
        pred_ok[t] = (fx - f_bq) - (-model_val)  # actual minus predicted reduction
        fp = problem.f_poised.flatten()
        contaminated[t] = np.max(fp) > 1e3 * max(np.min(np.abs(fp)), 1e-12)
        if pole_diag:
            pole_hit[t] = pole_in_box(x, r)

    edges = np.geomspace(*RADIUS_RANGE, N_BINS + 1)
    idx = np.clip(np.digitize(radii, edges) - 1, 0, N_BINS - 1)
    print(f"\n=== {label} ===")
    print("radius bin      median adv   bqmin win%   pole-hit%   contam%   "
          "median |pred err|")
    for b in range(N_BINS):
        m = idx == b
        if not m.any():
            continue
        print(f"[{edges[b]:7.3g},{edges[b+1]:7.3g})  "
              f"{np.median(adv[m]):11.4g}  "
              f"{100*np.mean(adv[m] > 0):9.1f}   "
              f"{100*np.mean(pole_hit[m]):8.1f}   "
              f"{100*np.mean(contaminated[m]):6.1f}   "
              f"{np.median(np.abs(pred_ok[m])):12.4g}")
    if pole_diag:
        for name, m in [("pole in box", pole_hit), ("no pole in box", ~pole_hit)]:
            if m.any():
                print(f"  {name:<15}: n={m.sum():4d}  median adv={np.median(adv[m]):10.4g}  "
                      f"bqmin win%={100*np.mean(adv[m] > 0):5.1f}")


# A: linear rank-1 residuals -> exactly quadratic objective
run(build_smooth_problem(np.zeros(3), m=15, nprob=2), "nprob=2 linear (f exactly quadratic)")
# B: Box 3D -- smooth, no poles
run(build_smooth_problem(np.zeros(3), m=15, nprob=12), "nprob=12 Box 3D (smooth, no poles)")
# C: Bard with pole diagnostics
run(build_smooth_problem(np.zeros(3), m=15, nprob=8), "nprob=8 Bard (poles)", pole_diag=True)
