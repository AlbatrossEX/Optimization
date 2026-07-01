import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from general_model.Trust_region_optimization import TR_function
from .algorism6_4 import algorithm_6_4
from .Quad import fitfroquad

Array1D = NDArray[np.floating]


class SmoothFunction(TR_function):
    def __init__(self, f):
        super().__init__(f)

    def GH(self, x: Array1D, radius: float) -> Tuple[Array1D, Array1D]:
        f_out: NDArray[np.floating] = np.array([[self.f(x)]])
        poised , _ = algorithm_6_4(Y=x.reshape(1, -1), Delta=radius, f=f_out)
        n = poised.shape[0]
        f_poised: NDArray[np.floating] = np.zeros((n, 1))
        for i in range(n):
            f_poised[i] = self.f(poised[i, :])

        return fitfroquad(poised, f_poised)
        
