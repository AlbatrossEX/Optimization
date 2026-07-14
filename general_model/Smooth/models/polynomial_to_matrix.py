import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def polynomial_to_matrix(
    L: NDArray[np.floating],
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], float]:
    b: float = float(L[0])
    c: NDArray[np.floating] = L[1 : n + 1]
    L = L[n + 1 :]
    Q: NDArray[np.floating] = np.zeros((n, n))

    for k in range(n):
        Q[k, k:] = L[:n - k]
        L = L[n - k:]

    D: NDArray[np.floating] = np.diag(Q)
    Q = Q - np.diag(D)
    Q = Q + Q.T + np.diag(D)
    return Q, c, b
