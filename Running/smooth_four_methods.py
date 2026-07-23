"""Objective 1 — Smooth function: compare the four trust-region methods.

Runs the smooth (differentiable) calfun problem with all four trust-region
solvers against the quadratic interpolation model (gh_type 0):
  0 = bqmin step
  1 = best interpolation point
  2 = better of the two
  3 = dynamic interpolation point

Everything about the experiment is declared here; general_model/Optimize.py owns
only how the run is executed. Radii concentrate the effort where trust-region
behaviour is most interesting: 90% of the sampled radii fall in [0.01, 1] and
10% in (1, 3] (see effort_radii). The stopping condition is a
function-evaluation budget (EVAL_BUDGET), not an iteration count.

Logs land in Log/Logs/<EVAL_BUDGET> Evalu/Smooth_four_methods/ — found or
created on the way in, so re-runs at the same budget share the directory and
replace same-named logs. Graph them with Log/Smooth_four_methods/graph.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from general_model.Optimize import (
    Experiment,
    build_cases,
    effort_radii,
    random_starts,
    run_experiment,
)

CONSTANTS = (
    0.1,   # miu
    0.1,   # theta
    0.5,   # shrink
    2.0,   # extend
    1.0,   # radius (single-run default; the suite varies radius per case)
    1.0,   # p
    0,     # method (single-run default; the suite varies method per case)
    0,     # gh_type (single-run default; the suite varies gh_type per case)
)

# Picklable problem spec; each parallel worker builds its own via build_problem.
PROBLEM = {"kind": "smooth", "m": 15, "nprob": 8}

EVAL_BUDGET = 2000  # function evaluations per run (the stopping condition)
# problem=PROBLEM rejects starts where the objective is not finite, so every
# run starts at a point the solver can rank against.
STARTS = random_starts(count=8, dim=3, box=3.0, seed=0, problem=PROBLEM)
# 90% of the radii in [0.01, 1], 10% in (1, 3].
RADII = effort_radii(count=20, low=0.01, mid=1.0, high=3.0, low_frac=0.9)

# All four trust-region methods, each on the interpolation model (gh_type 0).
COMBOS = [(0, 0), (1, 0), (2, 0), (3, 0)]

PREFIX = "S"

EXPERIMENT = Experiment(
    name="smooth_four_methods",
    problem=PROBLEM,
    constants=CONSTANTS,
    cases=build_cases(STARTS, RADII, COMBOS, PREFIX),
    evaluations=EVAL_BUDGET,
    prefix=PREFIX,
)

if __name__ == "__main__":
    run_experiment(EXPERIMENT)
