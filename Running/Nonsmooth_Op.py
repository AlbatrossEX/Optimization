"""Entry point for the non-smooth test suite: runs the non-differentiable
calfun problem over the same 10 starts x 10 radii grid as the smooth suite,
comparing the two GH model builders (gh_type 0 = quadratic interpolation fit,
gh_type 1 = random +-1 model) and the three trust-region methods.

Logs are labelled N### so they coexist with the smooth suite's T### logs in
Log/Logs; graph them with Log/Non_smooth/Nonsmooth_graph.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from general_model.Optimize import LOG_DIR, build_cases, describe_cases, run_suite

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

# (method, gh_type) pairs run for every start x radius: all three methods with
# the interpolation model, plus the random +-1 model — which only supports
# method 0, since it builds no interpolation set for methods 1/2 to step to.
COMBOS = [(0, 0), (1, 0), (2, 0), (0, 1)]

if __name__ == "__main__":
    cases = build_cases(combos=COMBOS, prefix="N")

    # Details and distribution of what is about to run: printed up front and
    # saved next to the log folder so it can be inspected after the run too.
    summary = describe_cases(cases)
    print(summary)
    (LOG_DIR.parent / "nonsmooth_case_distribution.txt").write_text(summary + "\n")

    # Wipe only this suite's own logs (N prefix, plus any live-log leftovers);
    # the smooth suite's T logs share the folder and must survive.
    for stale in LOG_DIR.glob("N*.txt"):
        stale.unlink()
    run_suite(PROBLEM, CONSTANTS, cases)
