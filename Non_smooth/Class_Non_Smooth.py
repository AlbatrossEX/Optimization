import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from general_model.Trust_region_optimization import TR_function


Array1D = NDArray[np.floating]


class NonSmoothFunction(TR_function):
    def __init__(self, f):
        super().__init__(f)

    def GH(self, x: Array1D, radius: float) -> Tuple[Array1D, Array1D]:
        n:int = x.shape[0]
        g: Array1D = np.zeros(n, dtype=float)
        h: Array1D = np.zeros((n, n), dtype=float)

        for i in range(n):
            g[i] = np.random.choice((-1, 1))

        for i in range(n):
            for j in range(i, n):
                h[i, j] = np.random.choice((-1, 1))
                h[j, i] = h[i, j]

        return g, h