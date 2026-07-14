import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def bqmin(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    L: NDArray[np.floating],
    U: NDArray[np.floating],
    tol: float = 1e-13,
    maxit: int = 5000,
) -> Tuple[NDArray[np.floating], float]:
    n: int = A.shape[0]
    X: NDArray[np.floating] = np.zeros(n)
    G: NDArray[np.floating] = A @ X + B
    Projg: NDArray[np.floating] = X - np.clip(X - G, L, U)
    it: int = 0

    while it < maxit and np.linalg.norm(Projg) > tol:
        it += 1
        t: float = 1.0
        pap: float = float(Projg @ A @ Projg)
        if pap > 0:
            t = min(1, (Projg @ G) / pap)
        X = X - t * Projg
        G = A @ X + B
        Projg = X - np.clip(X - G, L, U)

    f: float = float(0.5 * X @ A @ X + B @ X)
    return X, f
