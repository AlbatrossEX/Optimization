import numpy as np
from numpy.linalg import lstsq, qr
from numpy.typing import NDArray
from typing import Tuple


def phi2eval(X: NDArray[np.floating]) -> NDArray[np.floating]:
    X: NDArray[np.floating] = np.atleast_2d(X)
    m: int
    n: int
    m, n = X.shape
    num_terms: int = int(n * (n + 1) / 2)
    Phi: NDArray[np.floating] = np.zeros((m, num_terms))
    idx: int = 0
    for i in range(n):
        Phi[:, idx] = 0.5 * X[:, i] ** 2
        idx += 1
        for j in range(i + 1, n):
            Phi[:, idx] = X[:, i] * X[:, j] / np.sqrt(2)
            idx += 1
    return Phi


def fitfroquad(
    poised_set: NDArray[np.floating],
    f_poised: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    f_poised: NDArray[np.floating] = f_poised.flatten()
    m: int
    n: int
    m, n = poised_set.shape

    if m <= n + 1:
        M: NDArray[np.floating] = np.hstack((np.ones((m, 1)), poised_set))
        coeffs: NDArray[np.floating]
        coeffs, *_ = lstsq(M, f_poised, rcond=None)
        G: NDArray[np.floating] = coeffs[1:]
        H: NDArray[np.floating] = np.zeros((n, n))
        return G, H

    M = np.hstack((np.ones((m, 1)), poised_set))
    Q: NDArray[np.floating]
    Q, _ = qr(M, mode='reduced')
    Z: NDArray[np.floating] = np.eye(m) - Q @ Q.T

    Phi: NDArray[np.floating] = phi2eval(poised_set)
    N: NDArray[np.floating] = Phi.T
    L: NDArray[np.floating] = np.dot(N, Z)
    rhs: NDArray[np.floating] = np.dot(Z, f_poised)
    Beta: NDArray[np.floating]
    Beta, *_ = lstsq(L.T, rhs, rcond=None)

    residual: NDArray[np.floating] = f_poised - Phi @ Beta
    Alpha: NDArray[np.floating]
    Alpha, *_ = lstsq(M, residual, rcond=None)
    G: NDArray[np.floating] = Alpha[1:]

    H: NDArray[np.floating] = np.zeros((n, n))
    idx: int = 0
    for i in range(n):
        H[i, i] = Beta[idx]
        idx += 1
        for j in range(i + 1, n):
            H[i, j] = Beta[idx] / np.sqrt(2)
            H[j, i] = H[i, j]
            idx += 1

    return G, H
