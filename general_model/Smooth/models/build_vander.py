import numpy as np
from numpy.typing import NDArray
from typing import List


def build_vander(Y: NDArray[np.floating]) -> NDArray[np.floating]:
    m: int
    n: int
    m, n = Y.shape
    X: NDArray[np.floating] = np.copy(Y)
    vander_terms: List[NDArray[np.floating]] = []

    for i in range(n):
        for j in range(i, n):
            term = 0.5 * Y[:, i] * Y[:, j] if i == j else Y[:, i] * Y[:, j]
            vander_terms.append(term.reshape(-1, 1))

    quadratic_part: NDArray[np.floating] = np.hstack(vander_terms)
    return np.hstack((np.ones((m, 1)), X, quadratic_part))
