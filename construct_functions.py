import numpy as np
from numpy.typing import NDArray

from function.calfun import calfun
from general_model.Smooth.Class_Smooth import SmoothFunction
from general_model.Non_smooth.Class_Non_Smooth import NonSmoothFunction


Array1D = NDArray[np.floating]


def _calfun_objective(x: Array1D, m: int, nprob: int, probtype: str) -> np.float64:
    return np.float64(calfun(x, m, nprob, probtype=probtype, num_outs=1))


def build_smooth_problem(m: int, nprob: int = 1) -> SmoothFunction:
    return SmoothFunction(lambda v: _calfun_objective(v, m, nprob, "smooth"))


def build_nonsmooth_problem(m: int, nprob: int = 8) -> NonSmoothFunction:
    return NonSmoothFunction(lambda v: _calfun_objective(v, m, nprob, "nondiff"))


def build_problem(kind: str, m: int, nprob: int):
    """Build a problem from a picklable spec: kind is "smooth" or "nonsmooth".

    The problem objects themselves hold lambdas and a logger thread, so they
    cannot cross process boundaries; parallel workers rebuild one from a spec.
    """
    builders = {"smooth": build_smooth_problem, "nonsmooth": build_nonsmooth_problem}
    return builders[kind](m=m, nprob=nprob)
