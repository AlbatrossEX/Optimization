import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from construct_functions import Function_object
from general_model.Trust_region_optimization import TR_function
from pathlib import Path

Array1D = NDArray[np.floating]

CONSTANTS: Tuple[float, float, float, float, float, float] = (
    0.1,   # miu
    0.1,   # theta
    0.5,   # shrink
    1.1,   # extend
    10.0,  # radius
    1.0,   # p
)

LOG_DIR = Path("Log/Logs")
KEEP_LAST_RUNS = 5


def cleanup_logs(log_dir: Path = LOG_DIR, keep_last: int = KEEP_LAST_RUNS) -> None:
    """Drop crashed-run leftovers, empty logs, and all but the most recent runs."""
    orphan = log_dir / "New.txt"
    if orphan.exists():
        orphan.unlink()

    runs = [p for p in log_dir.glob("*.txt") if p.name != "New.txt"]
    for run in runs:
        if run.stat().st_size == 0:
            run.unlink()
    runs = [p for p in runs if p.exists()]

    runs.sort(key=lambda p: p.stat().st_mtime)
    for stale in runs[:-keep_last] if keep_last > 0 else runs:
        stale.unlink()


def main(
    x_0: Array1D,
    iteration: int,
) -> Array1D:
    miu, theta, shrink, extend, radius, p = CONSTANTS
    cleanup_logs()
    result = Function_object.trust_region_optimization(
        x_0=x_0,
        miu=miu,
        theta=theta,
        shrink=shrink,
        extend=extend,
        radius=radius,
        p=p,
        max_iter=iteration,
    )
    print(result)
    print(Function_object.output(result))
    Function_object.flush_log()
    log_name = f"miu{miu}_theta{theta}_shrink{shrink}_extend{extend}_radius{radius}_p{p}"
    Path("Log/Logs/New.txt").rename(f"Log/Logs/{log_name}.txt")
    cleanup_logs()
    return result


if __name__ == "__main__":
    x0: Array1D = np.array([1.0, 1.0, 1.0], dtype=float)
    main(x0, 200)
