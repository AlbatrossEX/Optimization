import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from construct_functions import Function_object
from general_model.Trust_region_optimization import TR_function
from datetime import datetime
from pathlib import Path

Array1D = NDArray[np.floating]

CONSTANTS: Tuple[float, float, float, float, float, float] = (
    0.1,   # miu
    1.0,   # theta
    0.8,   # shrink
    3.0,   # extend
    10.0,   # radius
    1.0,   # p
)

def main(
    x_0: Array1D,
    iteration: int,
) -> Array1D:
    miu, theta, shrink, extend, radius, p = CONSTANTS
    subject = Function_object
    result = subject.trust_region_optimization(
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
    print(subject.output(result))
    now = datetime.now()
    Path("Log/Logs/New.txt").rename(f"Log/Logs/{now}.txt")
    return result


if __name__ == "__main__":
    x0: Array1D = np.array([1.0, 1.0, 1.0], dtype=float)
    main(x0, 100)




