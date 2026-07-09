import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from general_model.Trust_region_optimization import TR_function
from Smooth.algorism6_4 import algorithm_6_4
from Smooth.Quad import fitfroquad


Array1D = NDArray[np.floating]


class NonSmoothFunction(TR_function):
    def GH(self, x: Array1D, radius: float, gh_type: int = 0) -> Tuple[Array1D, Array1D]:
        """gh_type 0 = quadratic interpolation fit (sets self.poised/self.f_poised);
        gh_type 1 = random +-1 model (no interpolation set, method 0 only)."""
        if gh_type == 0:
            f_out: NDArray[np.floating] = np.array([[self.output(x)]])
            poised, _ = algorithm_6_4(Y=x.reshape(1, -1), Delta=radius, f=f_out)
            n = poised.shape[0]
            f_poised: NDArray[np.floating] = np.zeros((n, 1))
            for i in range(n):
                f_poised[i] = self.output(poised[i, :])
            self.f_poised = f_poised
            self.poised = poised
            return fitfroquad(poised, f_poised)
        if gh_type == 1:
            n: int = x.shape[0]
            g: Array1D = np.random.choice((-1.0, 1.0), size=n)
            upper = np.triu(np.random.choice((-1.0, 1.0), size=(n, n)))
            h: Array1D = upper + np.triu(upper, 1).T
            return g, h
        raise ValueError(f"unknown gh_type {gh_type}")
