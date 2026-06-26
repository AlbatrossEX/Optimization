import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

from .bqmin import bqmin


Array1D = NDArray[np.floating]

CONSTANTS: Tuple[float, float, float, float, float, float] = (
    0.1,   # miu
    1.0,   # theta
    0.5,   # shrink
    2.0,   # extend
    1.0,   # radius
    1.0,   # p
)
EPSILON: float = 1e-6


def demo_f(x: Array1D) -> np.floating:
    return np.float64(x @ x)


def demo_GH(x: Array1D) -> Tuple[Array1D, Array1D]:
    g: Array1D = 2.0 * x
    h: Array1D = 2.0 * np.eye(x.shape[0])
    return g, h


class TR_function:
    def __init__(self, f: Callable[[Array1D], np.floating]):
        self.f = f

    def output(self, input: Array1D) -> np.floating:
        return self.f(input)

    def GH(self, x: Array1D) -> Tuple[Array1D, Array1D]:
        raise NotImplementedError

    def model(self, x: Array1D) -> Callable[[Array1D], np.floating]:
        g, h = self.GH(x)

        def f(step: Array1D) -> np.floating:
            return np.float64(self.f(x) + g @ step + 0.5 * step @ h @ step)

        return f

    def trust_region_optimization(
        self,
        x_0: Array1D,
        miu: float,
        theta: float,
        shrink: float,
        extend: float,
        radius: float,
        p: float,
        epsilon: float,
        max_iter: int = 1000,
    ) -> Array1D:
        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)

        for _ in range(max_iter):
            g, h = self.GH(x)
            grad_norm = float(np.linalg.norm(g, 2))
            if grad_norm <= epsilon or delta < epsilon:
                return x

            lower = -delta * np.ones_like(x)
            upper = delta * np.ones_like(x)
            step, _model_value = bqmin(h, g, lower, upper)
            step = np.asarray(step, dtype=float).reshape(x.shape)

            predicted_reduction = float(-(g @ step + 0.5 * step @ h @ step))
            if predicted_reduction <= 0:
                delta *= shrink
                continue

            actual_reduction = float(self.f(x) - self.f(x + step))
            roll = actual_reduction / (theta * (np.linalg.norm(step, 2) ** (1.0 + p)))

            if roll >= miu:
                x = x + step
                if roll > 0.75:
                    delta *= extend
            else:
                delta *= shrink

        return x


def main(
    x_0: Array1D,
    iteration: int,
) -> Array1D:
    miu, theta, shrink, extend, radius, p = CONSTANTS
    subject = TR_function(demo_f)
    subject.GH = demo_GH  # type: ignore[method-assign]
    result = subject.trust_region_optimization(
        x_0=x_0,
        miu=miu,
        theta=theta,
        shrink=shrink,
        extend=extend,
        radius=radius,
        p=p,
        epsilon=EPSILON,
        max_iter=iteration,
    )
    print(result)
    return result


if __name__ == "__main__":
    x0: Array1D = np.array([1.0, 1.0], dtype=float)
    main(x0, 1000)
