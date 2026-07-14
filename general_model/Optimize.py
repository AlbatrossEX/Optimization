import os
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, List, Tuple

from construct_functions import build_problem
from general_model.Trust_region_optimization import TR_function

Array1D = NDArray[np.floating]
# (miu, theta, shrink, extend, radius, p, method, gh_type) — defined by the entry
# point (Running/main.py) and passed in; nothing here hardcodes run parameters.
Constants = Tuple[float, float, float, float, float, float, int, int]

# anchored to the project root so runs work regardless of the caller's cwd
LOG_DIR = Path(__file__).resolve().parents[1] / "Log" / "Logs"
KEEP_LAST_RUNS = 5


def cleanup_logs(log_dir: Path = LOG_DIR, keep_last: int = KEEP_LAST_RUNS) -> None:
    """Drop crashed-run leftovers, empty logs, and all but the most recent runs."""
    # live/orphan logs: New.txt plus the per-worker New_<pid>.txt files
    for orphan in log_dir.glob("New*.txt"):
        orphan.unlink()

    runs = [p for p in log_dir.glob("*.txt") if not p.name.startswith("New")]
    for run in runs:
        if run.stat().st_size == 0:
            run.unlink()
    runs = [p for p in runs if p.exists()]

    runs.sort(key=lambda p: p.stat().st_mtime)
    for stale in runs[:-keep_last] if keep_last > 0 else runs:
        stale.unlink()


def log_name(label: str, x_0: Array1D, radius: float, p: float, method: int, gh_type: int) -> str:
    start = "-".join(f"{v:g}" for v in np.asarray(x_0, dtype=float))
    scenario = f"{label}_" if label else ""
    return f"{scenario}start{start}_radius{radius:g}_p{p:g}_method{method}_gh{gh_type}"


def main(
    problem: TR_function,
    constants: Constants,
    x_0: Array1D,
    iteration: int,
    radius: float = None,
    method: int = None,
    gh_type: int = None,
    label: str = "",
    keep_last: int = KEEP_LAST_RUNS,
) -> Array1D:
    """Single optimization run on an already-constructed problem object."""
    miu, theta, shrink, extend, radius_c, p, method_c, gh_type_c = constants
    if radius is None:
        radius = radius_c
    if method is None:
        method = method_c
    if gh_type is None:
        gh_type = gh_type_c

    cleanup_logs(keep_last=keep_last)
    result = problem.trust_region_optimization_function(
        method=method,
        x_0=x_0,
        miu=miu,
        theta=theta,
        shrink=shrink,
        extend=extend,
        radius=radius,
        p=p,
        max_iter=iteration,
        gh_type=gh_type,
    )
    print(result)
    print(problem.output(result))
    problem.flush_log()
    name = log_name(label, x_0, radius, p, method, gh_type)
    (LOG_DIR / "New.txt").rename(LOG_DIR / f"{name}.txt")
    cleanup_logs(keep_last=keep_last)
    return result


# --- Parallel test suite -----------------------------------------------------
# 10 radii (log-spaced 0.01 .. 10) x 10 random starting points x 3 solver
# methods (0 = bqmin step, 1 = best interpolation point, 2 = better of the two)
# = 300 optimization runs, executed on a process pool. Every case gets its own
# log file; the per-worker live log New_<pid>.txt keeps parallel writers from
# colliding on a shared New.txt.
N_RADII = 10
N_STARTS = 10
METHODS = (0, 1, 2)
ITERATION = 200
SEED = 0
START_BOX = 3.0  # starting points drawn uniformly from [-START_BOX, START_BOX]^3

Case = Tuple[str, Array1D, float, int, int]  # (label, x_0, radius, method, gh_type)

METHOD_NAMES = {0: "bqmin step", 1: "best interpolation point", 2: "better of the two"}


def describe_cases(cases: List[Case], iteration: int = ITERATION) -> str:
    """Human-readable details and distribution of a case set: which radii,
    methods, gh types, and starting points run, and how many runs each gets."""
    radii = Counter(float(c[2]) for c in cases)
    methods = Counter(int(c[3]) for c in cases)
    gh_types = Counter(int(c[4]) for c in cases)
    combos = Counter((int(c[3]), int(c[4])) for c in cases)
    starts: Dict[Tuple[float, ...], List] = {}  # start -> [n_runs, first_label, last_label]
    for label, x0, *_ in cases:
        entry = starts.setdefault(tuple(np.asarray(x0, dtype=float).tolist()), [0, label, label])
        entry[0] += 1
        entry[2] = label

    lines = [
        f"case set: {len(cases)} runs ({iteration} iterations each) = "
        f"{len(starts)} starting points x {len(radii)} radii x {len(combos)} method/gh combos",
        "  radii: " + ", ".join(f"{r:g} ({n} runs)" for r, n in sorted(radii.items())),
        "  methods: " + ", ".join(
            f"{m} = {METHOD_NAMES.get(m, 'unknown')} ({n} runs)" for m, n in sorted(methods.items())
        ),
        "  gh types: " + ", ".join(f"{g} ({n} runs)" for g, n in sorted(gh_types.items())),
        "  starting points:",
    ]
    for start, (n, first, last) in starts.items():
        point = "[" + ", ".join(f"{v:g}" for v in start) + "]"
        lines.append(f"    {first}..{last}  {point}  ({n} runs)")
    return "\n".join(lines)


def build_cases(
    gh_type: int = 0,
    combos: List[Tuple[int, int]] = None,
    prefix: str = "T",
) -> List[Case]:
    """combos are the (method, gh_type) pairs run for every start x radius;
    the default is every METHODS entry with the single gh_type."""
    if combos is None:
        combos = [(method, gh_type) for method in METHODS]
    # 3 significant figures keep the radius readable in the log filename while
    # still spanning 0.01 .. 10 log-uniformly.
    radii = [float(f"{r:.3g}") for r in np.logspace(np.log10(0.01), np.log10(10.0), N_RADII)]
    rng = np.random.default_rng(SEED)
    starts = np.round(rng.uniform(-START_BOX, START_BOX, size=(N_STARTS, 3)), 2)

    cases: List[Case] = []
    for x0 in starts:
        for radius in radii:
            for method, gh in combos:
                cases.append((f"{prefix}{len(cases):03d}", x0, radius, method, gh))
    return cases


# Per-worker state, set once by _init_worker: the problem object holds lambdas
# and a logger thread, so it cannot be pickled — each worker builds its own
# from the spec instead of receiving the parent's.
_PROBLEM: TR_function = None
_CONSTANTS: Constants = None


def _init_worker(problem_spec: Dict, constants: Constants) -> None:
    global _PROBLEM, _CONSTANTS
    _PROBLEM = build_problem(**problem_spec)
    _CONSTANTS = constants
    # One live log per worker process; renamed to the final name after each case.
    _PROBLEM.redirect_log(str(LOG_DIR / f"New_{os.getpid()}.txt"))


def run_case(case: Case, iteration: int = ITERATION) -> Tuple[str, str, float]:
    label, x0, radius, method, gh_type = case
    miu, theta, shrink, extend, _, p, _, _ = _CONSTANTS

    # A previous case that crashed in this worker leaves a stale live log
    # behind; drop it so its lines don't leak into this case's log.
    live = LOG_DIR / f"New_{os.getpid()}.txt"
    live.unlink(missing_ok=True)

    result = _PROBLEM.trust_region_optimization_function(
        method=method,
        x_0=x0,
        miu=miu,
        theta=theta,
        shrink=shrink,
        extend=extend,
        radius=radius,
        p=p,
        max_iter=iteration,
        gh_type=gh_type,
    )
    f_final = float(_PROBLEM.output(result))
    _PROBLEM.flush_log()

    name = log_name(label, x0, radius, p, method, gh_type)
    live.rename(LOG_DIR / f"{name}.txt")
    return label, name, f_final


def run_suite(
    problem_spec: Dict,
    constants: Constants,
    cases: List[Case],
    iteration: int = ITERATION,
    max_workers: int = None,
) -> None:
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(problem_spec, constants)
    ) as pool:
        futures = {pool.submit(run_case, case, iteration): case[0] for case in cases}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                label, name, f_final = future.result()
            except Exception as exc:  # one bad case should not kill the suite
                print(f"[{done}/{len(cases)}] {futures[future]}: FAILED ({exc})")
                continue
            print(f"[{done}/{len(cases)}] {label}: f = {f_final:.6g}  ({name})")
