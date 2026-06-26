import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from function.calfun import calfun
from Smooth.Class_Smooth import SmoothFunction
from Non-smooth.Class_Non_Smooth import NonSmoothFunction


Array1D = NDArray[np.floating]


def _calfun_objective(x: Array1D, m: int, nprob: int, probtype: str) -> np.float64:
    return np.float64(calfun(x, m, nprob, probtype=probtype, num_outs=1))


def _calfun_GH(
    x: Array1D,
    m: int,
    nprob: int,
    probtype: str,
) -> Tuple[Array1D, Array1D]:
    _, _, g, j = calfun(x, m, nprob, probtype=probtype, num_outs=4)
    return np.asarray(g, dtype=float), np.asarray(j, dtype=float)


def build_smooth_problem(
    x: Array1D,
    m: int,
    nprob: int = 1,
) -> SmoothFunction:
    subject = SmoothFunction(lambda v: _calfun_objective(v, m, nprob, "smooth"))

    def gh(v: Array1D) -> Tuple[Array1D, Array1D]:
        return _calfun_GH(v, m, nprob, "smooth")

    subject.GH = gh  # type: ignore[method-assign]
    return subject


def build_nonsmooth_problem(
    x: Array1D,
    m: int,
    nprob: int = 8,
) -> NonSmoothFunction:
    subject = NonSmoothFunction(lambda v: _calfun_objective(v, m, nprob, "nondiff"))

    def gh(v: Array1D) -> Tuple[Array1D, Array1D]:
        return _calfun_GH(v, m, nprob, "nondiff")

    subject.GH = gh  # type: ignore[method-assign]
    return subject


def main() -> None:
    x_smooth: Array1D = np.asarray([0.1, 0.2, 0.3], dtype=float)
    x_nonsmooth: Array1D = np.asarray([0.1, 0.2, 0.3], dtype=float)

    smooth_obj = build_smooth_problem(x_smooth, m=3, nprob=4)
    nonsmooth_obj = build_nonsmooth_problem(x_nonsmooth, m=15, nprob=8)

    print("smooth:", smooth_obj.output(x_smooth))
    print("nonsmooth:", nonsmooth_obj.output(x_nonsmooth))


if __name__ == "__main__":
    main()
