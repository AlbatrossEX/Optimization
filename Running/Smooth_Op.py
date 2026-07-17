"""Research objective (baseline) — Smooth function: the original three-method
comparison over the full radius decade.

The historical smooth suite: the smooth (differentiable) calfun problem solved by
the three original trust-region methods (0 = bqmin step, 1 = best interpolation
point, 2 = better of the two) against the quadratic interpolation model, with the
radius swept log-uniformly across the whole [0.01, 10] decade. It is kept as the
broad-sweep baseline; the focused four-method study (which adds method 3 and
concentrates 90% of the radii in [0.01, 1]) lives in Running/smooth_four_methods.py.

Everything about the experiment is declared here; general_model/Optimize.py owns
only how the run is executed. The stopping condition is a function-evaluation
budget (EVAL_BUDGET). Logs land in their own Log/Logs/smooth_op_<timestamp>/
directory and are never cleared; the archived legacy runs live in
Log/Logs/smooth_op_legacy_*/. Graph them with Log/BQmin_graphing/BQ_algo_compare.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from general_model.Optimize import (
    Experiment,
    build_cases,
    log_radii,
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

EVAL_BUDGET = 10000  # function evaluations per run (the stopping condition)
STARTS = random_starts(count=10, dim=3, box=3.0, seed=0)
# Broad baseline sweep: log-uniform across the whole [0.01, 10] radius decade.
RADII = log_radii(0.01, 10.0, 10)

# The three original trust-region methods against the interpolation model (gh 0).
COMBOS = [(0, 0), (1, 0), (2, 0)]

PREFIX = "T"

EXPERIMENT = Experiment(
    name="smooth_op",
    problem=PROBLEM,
    constants=CONSTANTS,
    cases=build_cases(STARTS, RADII, COMBOS, PREFIX),
    evaluations=EVAL_BUDGET,
    prefix=PREFIX,
)

if __name__ == "__main__":
    run_experiment(EXPERIMENT)
