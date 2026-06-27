import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from general_model.Trust_region_optimization import TR_function


Array1D = NDArray[np.floating]


class NonSmoothFunction(TR_function):
    def __init__(self, f):
        super().__init__(f)

    def GH(self, x: Array1D) -> Tuple[Array1D, Array1D]:
        raise NotImplementedError
