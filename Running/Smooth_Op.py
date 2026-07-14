"""Entry point: defines the run configuration (constants + problem construction)
and launches the parallel trust-region test suite from general_model/Optimize.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from general_model.Optimize import LOG_DIR, build_cases, describe_cases, run_suite

CONSTANTS = (
    0.1,   # miu
    0.1,   # theta
    0.5,   # shrink
    1.5,   # extend
    1.0,  # radius (single-run default; the suite varies radius per case)
    1.0,   # p
    2,     # method: 0 = bqmin step, 1 = best interpolation point, 2 = better of the two
    0,     # gh_type: model builder inside GH (0 = quadratic interpolation fit)
)

# Function construction: a picklable spec (kind is "smooth" or "nonsmooth");
# each parallel worker builds its own problem via construct_functions.build_problem.
PROBLEM = {"kind": "smooth", "m": 15, "nprob": 8}

if __name__ == "__main__":
    cases = build_cases(gh_type=CONSTANTS[7])

    # Details and distribution of what is about to run: printed up front and
    # saved next to the log folder so it can be inspected after the run too.
    summary = describe_cases(cases)
    print(summary)
    (LOG_DIR.parent / "case_distribution.txt").write_text(summary + "\n")

    # Start from a clean slate so the suite ends with exactly len(cases) logs.
    for stale in LOG_DIR.glob("*.txt"):
        stale.unlink()
    run_suite(PROBLEM, CONSTANTS, cases)
