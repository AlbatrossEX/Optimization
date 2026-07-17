"""Objective 3 — Non-smooth function: compare method 0 (GH 0) vs methods 1, 2, 3.

Runs the non-differentiable calfun problem with all four trust-region solvers on
the quadratic interpolation model (gh_type 0), so method 0 (bqmin step) can be
read against the three interpolation-point strategies:
  0 = bqmin step
  1 = best interpolation point
  2 = better of the two
  3 = dynamic interpolation point

All four share gh_type 0 (the interpolation model), which is the only model the
interpolation-point methods can use. Radii concentrate the effort where
trust-region behaviour is most interesting: 90% in [0.01, 1], 10% in (1, 3] (see
effort_radii). The stopping condition is a function-evaluation budget
(EVAL_BUDGET).

Logs land in their own Log/Logs/nonsmooth_method_compare_<timestamp>/ directory
and are never cleared. Graph them with Log/Nonsmooth_method_compare/graph.py.
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
    1.5,   # extend
    1.0,   # radius (single-run default; the suite varies radius per case)
    1.0,   # p
    0,     # method (single-run default; the suite varies method per case)
    0,     # gh_type (single-run default; the suite varies gh_type per case)
)

PROBLEM = {"kind": "nonsmooth", "m": 15, "nprob": 8}

EVAL_BUDGET = 10000  # function evaluations per run (the stopping condition)
STARTS = random_starts(count=8, dim=3, box=3.0, seed=0)
# 90% of the radii in [0.01, 1], 10% in (1, 3].
RADII = effort_radii(count=20, low=0.01, mid=1.0, high=3.0, low_frac=0.9)

# Method 0 (bqmin step, gh 0) against the three interpolation-point methods,
# all on the interpolation model.
COMBOS = [(0, 0), (1, 0), (2, 0), (3, 0)]

PREFIX = "M"

EXPERIMENT = Experiment(
    name="nonsmooth_method_compare",
    problem=PROBLEM,
    constants=CONSTANTS,
    cases=build_cases(STARTS, RADII, COMBOS, PREFIX),
    evaluations=EVAL_BUDGET,
    prefix=PREFIX,
)

if __name__ == "__main__":
    run_experiment(EXPERIMENT)
