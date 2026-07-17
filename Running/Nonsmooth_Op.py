"""Research objective (baseline) — Non-smooth function: the original combined
suite over the full radius decade.

The historical non-smooth suite, kept as the broad-sweep baseline. It bundles two
comparisons into one grid on the non-differentiable calfun problem, radius swept
log-uniformly across the whole [0.01, 10] decade:
  - GH_COMPARE    : method 0 (bqmin step) with gh 0 (interpolation fit) vs gh 1
                    (random +-1 model) -- a pure model-builder comparison.
  - METHOD_COMPARE: methods 1 and 2 on gh 0 -- context for the above, so the GH
                    gap can be read against how the interpolation-point methods
                    themselves differ.

Method 0 is the only method the GH comparison can use: gh 1 builds no
interpolation set, and methods 1/2/3 step to an interpolation point (see
Class_Non_Smooth.GH). The focused single-objective studies (concentrating 90% of
the radii in [0.01, 1], and adding method 3) live in
Running/nonsmooth_gh_compare.py and Running/nonsmooth_method_compare.py.

Everything about the experiment is declared here; general_model/Optimize.py owns
only how the run is executed. The stopping condition is a function-evaluation
budget (EVAL_BUDGET). Logs land in their own Log/Logs/nonsmooth_op_<timestamp>/
directory and are never cleared; the archived legacy runs live in
Log/Logs/nonsmooth_op_legacy_*/. Graph them with the scripts in Log/Non_smooth/.
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
    0,     # method (single-run default; the suite varies method per case)
    0,     # gh_type (single-run default; the suite varies gh_type per case)
)

PROBLEM = {"kind": "nonsmooth", "m": 15, "nprob": 8}

EVAL_BUDGET = 2000  # function evaluations per run (the stopping condition)
STARTS = random_starts(count=10, dim=3, box=3.0, seed=0)
# Broad baseline sweep: log-uniform across the whole [0.01, 10] radius decade.
RADII = log_radii(0.01, 10.0, 10)

# Model-builder comparison: one method, both GH builders, everything else fixed.
GH_COMPARE = Block(methods=[0], gh_types=[0, 1], starts=STARTS, radii=RADII)
# The other interpolation-point methods, so the GH gap reads against how the
# methods themselves differ. Its own block because it cannot include gh 1.
METHOD_COMPARE = Block(methods=[1, 2], gh_types=[0], starts=STARTS, radii=RADII)
BLOCKS = [GH_COMPARE, METHOD_COMPARE]

PREFIX = "N"

EXPERIMENT = Experiment(
    name="nonsmooth_op",
    problem=PROBLEM,
    constants=CONSTANTS,
    cases=build_blocks(BLOCKS, PREFIX),
    evaluations=EVAL_BUDGET,
    prefix=PREFIX,
)

if __name__ == "__main__":
    run_experiment(EXPERIMENT)
