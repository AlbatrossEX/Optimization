import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.typing import NDArray
from typing import Callable, Dict, List, Tuple
from construct_functions import Function_object
from general_model.Trust_region_optimization import TR_function
from pathlib import Path

Array1D = NDArray[np.floating]

CONSTANTS: Tuple[float, float, float, float, float, float, int, int] = (
    0.1,   # miu
    0.1,   # theta
    0.5,   # shrink
    1.1,   # extend
    10.0,  # radius
    1.0,   # p
    2,     # method: 0 = bqmin step, 1 = best interpolation point, 2 = better of the two
    0,     # gh_type: model builder inside GH (0 = quadratic interpolation fit)
)

LOG_DIR = Path("Log/Logs")
KEEP_LAST_RUNS = 5


def cleanup_logs(log_dir: Path = LOG_DIR, keep_last: int = KEEP_LAST_RUNS) -> None:
    """Drop crashed-run leftovers, empty logs, and all but the most recent runs."""
    # live/orphan logs: New.txt plus the per-worker New_<pid>.txt files
    for orphan in log_dir.glob("New*.txt"):
        orphan.unlink()

    runs = [p for p in log_dir.glob("*.txt") if not p.name.startswith("New") and p.name != "classification.txt"]
    for run in runs:
        if run.stat().st_size == 0:
            run.unlink()
    runs = [p for p in runs if p.exists()]

    runs.sort(key=lambda p: p.stat().st_mtime)
    for stale in runs[:-keep_last] if keep_last > 0 else runs:
        stale.unlink()


LOG_LINE = re.compile(r"^(\d+),\[(.*?)\],([-+\deE.]+),\s*$")


def classify_logs(log_dir: Path = LOG_DIR, summary_name: str = "classification.txt") -> Dict[str, str]:
    """Classify each finished log by convergence behavior and write a summary.

    Labels:
      converged  - plateaued after reducing f by >= 99% (or reaching ~0)
      stalled    - plateaued while a substantial part of the initial f remains
      improving  - still making progress when the iteration budget ran out
      diverged   - never improved on the starting objective
    """
    labels: Dict[str, str] = {}
    for log in sorted(log_dir.glob("*.txt")):
        if log.name.startswith("New") or log.name == summary_name:
            continue
        objectives = [
            float(m.group(3))
            for m in (LOG_LINE.match(line) for line in log.read_text().splitlines())
            if m
        ]
        if not objectives:
            continue

        best = np.minimum.accumulate(np.array(objectives))
        initial, final = best[0], best[-1]
        reduction = (initial - final) / max(abs(initial), 1e-300)
        # Plateaued when the best-so-far value barely moved over the last
        # quarter of the run (relative to the value it settled at).
        tail = best[-max(len(best) // 4, 2):]
        plateaued = (tail[0] - tail[-1]) <= 1e-6 * max(abs(tail[-1]), 1.0)

        if final >= initial:
            labels[log.name] = "diverged"
        elif not plateaued:
            labels[log.name] = "improving"
        elif reduction >= 0.99 or abs(final) < 1e-8:
            labels[log.name] = "converged"
        else:
            labels[log.name] = "stalled"

    lines = [f"{label:<10} {name}" for name, label in sorted(labels.items(), key=lambda kv: (kv[1], kv[0]))]
    (log_dir / summary_name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return labels


def log_name(label: str, x_0: Array1D, radius: float, p: float, method: int, gh_type: int) -> str:
    start = "-".join(f"{v:g}" for v in np.asarray(x_0, dtype=float))
    scenario = f"{label}_" if label else ""
    return f"{scenario}start{start}_radius{radius:g}_p{p:g}_method{method}_gh{gh_type}"


def main(
    x_0: Array1D,
    iteration: int,
    radius: float = None,
    method: int = None,
    gh_type: int = None,
    label: str = "",
    keep_last: int = KEEP_LAST_RUNS,
) -> Array1D:
    miu, theta, shrink, extend, radius_c, p, method_c, gh_type_c = CONSTANTS
    if radius is None:
        radius = radius_c
    if method is None:
        method = method_c
    if gh_type is None:
        gh_type = gh_type_c

    cleanup_logs(keep_last=keep_last)
    result = Function_object.trust_region_optimization_function(
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
    print(Function_object.output(result))
    Function_object.flush_log()
    name = log_name(label, x_0, radius, p, method, gh_type)
    Path("Log/Logs/New.txt").rename(f"Log/Logs/{name}.txt")
    cleanup_logs(keep_last=keep_last)
    classify_logs()
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


def build_cases() -> List[Case]:
    # 3 significant figures keep the radius readable in the log filename while
    # still spanning 0.01 .. 10 log-uniformly.
    radii = [float(f"{r:.3g}") for r in np.logspace(np.log10(0.01), np.log10(10.0), N_RADII)]
    rng = np.random.default_rng(SEED)
    starts = np.round(rng.uniform(-START_BOX, START_BOX, size=(N_STARTS, 3)), 2)

    _, _, _, _, _, _, _, gh_type = CONSTANTS
    cases: List[Case] = []
    for x0 in starts:
        for radius in radii:
            for method in METHODS:
                cases.append((f"T{len(cases):03d}", x0, radius, method, gh_type))
    return cases


def _init_worker() -> None:
    # One live log per worker process; renamed to the final name after each case.
    Function_object.redirect_log(str(LOG_DIR / f"New_{os.getpid()}.txt"))


def run_case(case: Case, iteration: int = ITERATION) -> Tuple[str, str, float]:
    label, x0, radius, method, gh_type = case
    miu, theta, shrink, extend, _, p, _, _ = CONSTANTS

    # A previous case that crashed in this worker leaves a stale live log
    # behind; drop it so its lines don't leak into this case's log.
    live = LOG_DIR / f"New_{os.getpid()}.txt"
    live.unlink(missing_ok=True)

    result = Function_object.trust_region_optimization_function(
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
    f_final = float(Function_object.output(result))
    Function_object.flush_log()

    name = log_name(label, x0, radius, p, method, gh_type)
    live.rename(LOG_DIR / f"{name}.txt")
    return label, name, f_final


def run_suite(cases: List[Case], iteration: int = ITERATION, max_workers: int = None) -> None:
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as pool:
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
    classify_logs()


if __name__ == "__main__":
    cases = build_cases()
    # Start from a clean slate so the suite ends with exactly len(cases) logs.
    for stale in LOG_DIR.glob("*.txt"):
        stale.unlink()
    run_suite(cases)
