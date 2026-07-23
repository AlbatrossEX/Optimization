import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

from general_model.Trust_region_optimization import TR_function
from general_model.Smooth.algorism6_4 import algorithm_6_4
from general_model.Smooth.Quad import fitfroquad


Array1D = NDArray[np.floating]


class NonSmoothFunction(TR_function):
    def _predicted_reduction(
        self,
        step: Array1D,
        g: Optional[Array1D],
        h: Optional[Array1D],
        theta: float,
        p: float,
    ) -> float:
        """Basic DFO-TRNS ratio denominator: the forcing term theta*||s||^(1+p),
        used in place of a model reduction so that

            rho = [f(x) - f(x+s)] / [theta*||s||^(1+p)].

        On unsuccessful steps this makes the predicted reduction behave like
        o(||s||), which the nonsmooth convergence analysis needs to prove the
        Clarke generalized derivative is nonnegative along the relevant
        directions. The model (g, h) is deliberately not used here.
        """
        return theta * float(np.linalg.norm(step, 2)) ** (1.0 + p)

    def GH(self, x: Array1D, radius: float, gh_type: int ) -> Tuple[Array1D, Array1D]:
        """gh_type 0 = quadratic interpolation fit (sets self.poised/self.f_poised);
        gh_type 1 = random +-1 model (no interpolation set, method 0 only)."""
        if gh_type == 0:
            f_out: NDArray[np.floating] = np.array([[self.output(x)]])
            offsets, _ = algorithm_6_4(Y=np.zeros_like(x.reshape(1, -1)), Delta=radius, f=f_out)
            poised = offsets + x
            # price the whole poised set at once (optionally multi-threaded; see
            # TR_function._evaluate_poised) and shape it as a column
            f_poised: NDArray[np.floating] = self._evaluate_poised(poised).reshape(-1, 1)
            self.f_poised = f_poised
            self.poised = poised
            # fit on the offsets so g is the model gradient at x (matching
            # bqmin's step box), not the coefficient at the coordinate origin
            return fitfroquad(offsets, f_poised)
        if gh_type == 1:
            n: int = x.shape[0]
            g: Array1D = np.random.choice((-1.0, 1.0), size=n)
            upper = np.triu(np.random.choice((-1.0, 1.0), size=(n, n)))
            h: Array1D = upper + np.triu(upper, 1).T
            return g, h
        raise ValueError(f"unknown gh_type {gh_type}")
