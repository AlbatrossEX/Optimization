import numpy as np
import importlib.util
from pathlib import Path
from numpy.typing import NDArray
from typing import Tuple

from function.calfun import calfun
from Smooth.Class_Smooth import SmoothFunction

_NONSMOOTH_PATH = Path(__file__).parent / "Non-smooth" / "Class_Non_Smooth.py"
_NONSMOOTH_SPEC = importlib.util.spec_from_file_location(
    "Class_Non_Smooth",
    _NONSMOOTH_PATH,
)
if _NONSMOOTH_SPEC is None or _NONSMOOTH_SPEC.loader is None:
    raise ImportError(f"Cannot load NonSmoothFunction from {_NONSMOOTH_PATH}")
_NONSMOOTH_MODULE = importlib.util.module_from_spec(_NONSMOOTH_SPEC)
_NONSMOOTH_SPEC.loader.exec_module(_NONSMOOTH_MODULE)
NonSmoothFunction = _NONSMOOTH_MODULE.NonSmoothFunction


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


Function_object = build_nonsmooth_problem(
    np.asarray([0.1, 0.2, 0.3], dtype=float),
    m=15,
    nprob=8,
)
