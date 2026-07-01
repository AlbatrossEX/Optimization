import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from .bqmin import bqmin

Array1D = NDArray[np.floating]


def demo_f(x: Array1D) -> np.floating:
    return np.float64(x @ x)


def demo_GH(x: Array1D) -> Tuple[Array1D, Array1D]:
    g: Array1D = 2.0 * x
    h: Array1D = 2.0 * np.eye(x.shape[0])
    return g, h


class TR_function:
    def __init__(self, f: Callable[[Array1D], np.floating]):
        self.f = f
        self.count = 0

    def output(self, input: Array1D) -> np.floating:
        self.count += 1
        with open("Log/Logs/New.txt", "a") as f:
            f.write(f"{self.count},{input},{self.f(input)},\n")
        
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
        max_iter: int = 1000,
    ) -> Array1D:
        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)

        for _ in range(max_iter):
            g, h = self.GH(x,delta)
            lower = -delta * np.ones_like(x)
            upper = delta * np.ones_like(x)
            step, _ = bqmin(h, g, lower, upper)
            step = np.asarray(step, dtype=float).reshape(x.shape)

            predicted_reduction = float(-(g @ step + 0.5 * step @ h @ step))
            if predicted_reduction <= 0:
                delta *= shrink
                continue

            actual_reduction = float(self.f(x) - self.f(x + step))
            if not np.isfinite(actual_reduction):
                delta *= shrink
                continue
            roll = actual_reduction / (theta * (np.linalg.norm(step, 2) ** (1.0 + p)))

            if roll >= miu:
                x = x + step
                if roll > 0.75:
                    delta *= extend
            else:
                delta *= shrink

        return x

