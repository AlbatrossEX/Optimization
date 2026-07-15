"""Entry point for the smooth test suite.

Everything about the experiment is declared here — the function, constants,
starting points, radii, gh types, trust-region methods, iteration count and case
distribution. general_model/Optimize.py owns only how the run is executed.
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
    2,     # method (single-run default; the suite varies method per case)
    0,     # gh_type (single-run default; the suite varies gh_type per case)
)

# Function construction: a picklable spec (kind is "smooth" or "nonsmooth");
# each parallel worker builds its own problem via construct_functions.build_problem.
PROBLEM = {"kind": "smooth", "m": 15, "nprob": 8}

ITERATION = 2000
STARTS = random_starts(count=10, dim=3, box=3.0, seed=0)
RADII = log_radii(0.01, 10.0, 10)

# (method, gh_type) pairs run for every start x radius: all three trust-region
# methods against the quadratic interpolation model.
COMBOS = [(0, 0), (1, 0), (2, 0)]

# Label prefix for this suite's logs; the non-smooth suite uses "N" and the two
# coexist in Log/Logs. Graph these with Log/Graph.py.
PREFIX = "T"

if __name__ == "__main__":
    run_experiment(
        Experiment(
            name="smooth",
            problem=PROBLEM,
            constants=CONSTANTS,
            cases=build_cases(STARTS, RADII, COMBOS, PREFIX),
            iteration=ITERATION,
            prefix=PREFIX,
        )
    )
