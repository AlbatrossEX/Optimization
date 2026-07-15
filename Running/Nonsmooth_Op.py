"""Entry point for the non-smooth test suite: the non-differentiable calfun
problem, built to compare the two GH model builders (gh_type 0 = quadratic
interpolation fit, gh_type 1 = random +-1 model).

Everything about the experiment is declared here — the function, constants,
starting points, radii, gh types, trust-region methods, iteration count and case
distribution. general_model/Optimize.py owns only how the run is executed.

The distribution is declared as Blocks rather than one uniform grid, so each
comparison can be scoped independently. Logs are labelled N### and coexist with
the smooth suite's T### logs in Log/Logs; graph them with
Log/Non_smooth/Nonsmooth_graph.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from general_model.Optimize import (
    Block,
    Experiment,
    build_blocks,
    log_radii,
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
    2,     # method (single-run default; the suite varies method per case)
    0,     # gh_type (single-run default; the suite varies gh_type per case)
)

# Function construction: a picklable spec; each parallel worker builds its own
# problem via construct_functions.build_problem.
PROBLEM = {"kind": "nonsmooth", "m": 15, "nprob": 8}

ITERATION = 500
STARTS = random_starts(count=10, dim=3, box=3.0, seed=0)
RADII = log_radii(0.01, 10.0, 10)

# The comparison this suite exists for: one trust-region method, both GH model
# builders, everything else held fixed — so any difference in the logs is the
# model builder and nothing else.
#
# Method 0 (bqmin step) is the only method this comparison can run at: gh_type 1
# builds no interpolation set, and methods 1/2 step to the best interpolation
# point, so they have nothing to step to. See Class_Non_Smooth.GH.
GH_COMPARE = Block(methods=[0], gh_types=[0, 1], starts=STARTS, radii=RADII)

# Context for the above: the other two trust-region methods against the
# interpolation model, so the GH comparison can be read against how the methods
# themselves differ. Its own block because it cannot include gh_type 1.
METHOD_COMPARE = Block(methods=[1, 2], gh_types=[0], starts=STARTS, radii=RADII)

# Each Block carries its own starts and radii, so a comparison can be scoped
# independently of the others — e.g. sweep the GH comparison over a denser grid
# with radii=log_radii(0.01, 10.0, 20), or repeat a block to average over the
# random draw gh_type 1 makes. Blocks are concatenated; labels stay unique.
BLOCKS = [GH_COMPARE, METHOD_COMPARE]

# Label prefix for this suite's logs; the smooth suite uses "T" and the two
# coexist in Log/Logs.
PREFIX = "N"

if __name__ == "__main__":
    run_experiment(
        Experiment(
            name="nonsmooth",
            problem=PROBLEM,
            constants=CONSTANTS,
            cases=build_blocks(BLOCKS, PREFIX),
            iteration=ITERATION,
            prefix=PREFIX,
        )
    )
