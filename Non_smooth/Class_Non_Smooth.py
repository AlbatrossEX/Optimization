import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from general_model.Trust_region_optimization import TR_function


Array1D = NDArray[np.floating]


class NonSmoothFunction(TR_function):
    def GH(self, x: Array1D, radius: float) -> Tuple[Array1D, Array1D]:
        n: int = x.shape[0]
        g: Array1D = np.random.choice((-1.0, 1.0), size=n)
        upper = np.triu(np.random.choice((-1.0, 1.0), size=(n, n)))
        h: Array1D = upper + np.triu(upper, 1).T
        return g, h