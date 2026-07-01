import numpy as np
import importlib.util
from pathlib import Path
from numpy.typing import NDArray
from typing import Tuple

from function.calfun import calfun
from Smooth.Class_Smooth import SmoothFunction
from Non_smooth.Class_Non_Smooth import NonSmoothFunction


Array1D = NDArray[np.floating]


def _calfun_objective(x: Array1D, m: int, nprob: int, probtype: str) -> np.float64:
    return np.float64(calfun(x, m, nprob, probtype=probtype, num_outs=1))


def build_smooth_problem(
    x: Array1D,
    m: int,
    nprob: int = 1,
) -> SmoothFunction:
    subject = SmoothFunction(lambda v: _calfun_objective(v, m, nprob, "smooth"))
    return subject


def build_nonsmooth_problem(
    x: Array1D,
    m: int,
    nprob: int = 8,
) -> NonSmoothFunction:
    subject = NonSmoothFunction(lambda v: _calfun_objective(v, m, nprob, "nondiff"))
    return subject


Smooth_object = build_smooth_problem(
    np.asarray([0.1, 0.2, 0.3], dtype=float),
    m=15,
    nprob=8,
)

Non_Smooth_object = build_nonsmooth_problem(
    np.asarray([0.1, 0.2, 0.3], dtype=float),
    m=15,
    nprob=8,
)   
