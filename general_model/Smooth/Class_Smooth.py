import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

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

    def _predicted_reduction(
        self,
        step: Array1D,
        g: Optional[Array1D],
        h: Optional[Array1D],
        theta: float,
        p: float,
    ) -> float:
        """MBTR (Algorithm 11.1) ratio denominator: the model's OWN predicted
        decrease. For f ~ C^1 the quadratic model is f~(x+s) = f(x) + g.s +
        1/2 s.H.s, so the model reduction from s = 0 to s = step is

            f~(x) - f~(x+step) = -(g.step + 1/2 step.H.step).

        This is what makes rho = [f(x) - f(x+s)] / [f~(x) - f~(x+s)] measure how
        well the model predicted the true decrease, as MBTR requires.

        Methods 1 and 3 fit no model (g and h are None); with no model there is
        no model reduction, so they fall back to the same forcing term the
        nonsmooth (DFO-TRNS) case uses, which needs only the step length.
        """
        if g is None or h is None:
            return theta * float(np.linalg.norm(step, 2)) ** (1.0 + p)
        g = np.asarray(g, dtype=float)
        h = np.asarray(h, dtype=float)
        return float(-(g @ step + 0.5 * step @ h @ step))

