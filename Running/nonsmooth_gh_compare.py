"""Objective 2 — Non-smooth function: compare method 0 using GH 0 vs GH 1.

Runs the non-differentiable calfun problem with a single trust-region method
(0 = bqmin step) driven by the two GH model builders, everything else held
fixed, so any difference in the logs is the model builder and nothing else:
  gh_type 0 = quadratic interpolation fit
  gh_type 1 = random +-1 model

Method 0 is the only method this comparison can use: gh_type 1 builds no
interpolation set, and methods 1/2/3 step to an interpolation point, so they
have nothing to step to. See Class_Non_Smooth.GH.

Radii concentrate the effort where trust-region behaviour is most interesting:
90% in [0.01, 1], 10% in (1, 3] (see effort_radii). The stopping condition is a
function-evaluation budget (EVAL_BUDGET). gh_type 1 draws a random model, so its
block is repeated to average over that draw.

Logs land in Log/Logs/<EVAL_BUDGET> Evalu/Nonsmooth_gh_compare/ — found or
created on the way in, so re-runs at the same budget share the directory and
replace same-named logs. Graph them with Log/Nonsmooth_gh_compare/graph.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from general_model.Optimize import (
    Block,
    Experiment,
    build_blocks,
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

# The comparison this suite exists for: method 0, both GH model builders. gh 1's
# model is random, so the block is listed twice to average over two draws;
# gh 0 is deterministic, so its identical repeat only adds robustness to the
# per-case win comparisons. Blocks are concatenated; labels stay unique.
GH0 = Block(methods=[0], gh_types=[0], starts=STARTS, radii=RADII)
GH1_DRAW_A = Block(methods=[0], gh_types=[1], starts=STARTS, radii=RADII)
GH1_DRAW_B = Block(methods=[0], gh_types=[1], starts=STARTS, radii=RADII)
BLOCKS = [GH0, GH1_DRAW_A, GH1_DRAW_B]

PREFIX = "G"

EXPERIMENT = Experiment(
    name="nonsmooth_gh_compare",
    problem=PROBLEM,
    constants=CONSTANTS,
    cases=build_blocks(BLOCKS, PREFIX),
    evaluations=EVAL_BUDGET,
    prefix=PREFIX,
)

if __name__ == "__main__":
    run_experiment(EXPERIMENT)
