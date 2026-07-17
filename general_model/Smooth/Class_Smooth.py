import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from general_model.Trust_region_optimization import TR_function
from .algorism6_4 import algorithm_6_4
from .Quad import fitfroquad

Array1D = NDArray[np.floating]


class SmoothFunction(TR_function):
    def GH(self, x: Array1D, radius: float, gh_type: int = 0) -> Tuple[Array1D, Array1D]:
        """gh_type 0 = quadratic interpolation fit (the only smooth model builder)."""
        if gh_type != 0:
            raise ValueError(f"unknown gh_type {gh_type}")
        f_out: NDArray[np.floating] = np.array([[self.output(x)]])
        offsets,_ = algorithm_6_4(Y=np.zeros_like(x.reshape(1, -1)), Delta=radius, f=f_out)
        poised = offsets + x
        # price the whole poised set at once (optionally multi-threaded; see
        # TR_function._evaluate_poised) and shape it as a column
        f_poised: NDArray[np.floating] = self._evaluate_poised(poised).reshape(-1, 1)
        self.f_poised = f_poised
        self.poised = poised
        # fit on the offsets so g is the model gradient at x (matching
        # bqmin's step box), not the coefficient at the coordinate origin
        return fitfroquad(offsets, f_poised)
        
