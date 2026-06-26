import numpy as np
from scipy.linalg import eigh
from numpy.typing import NDArray
from typing import Callable, Tuple


def seceqn(
    lambda_: NDArray[np.floating],
    eigval: NDArray[np.floating],
    alpha: NDArray[np.floating],
    delta: float,
) -> NDArray[np.floating]:
    lambda_ = np.atleast_1d(lambda_)
    value: list[float] = []

    for lam in lambda_:
        denom = eigval + lam
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(denom != 0, (alpha / denom) ** 2, np.inf)
        s_norm = np.sqrt(np.sum(term))
        if np.isnan(s_norm):
            s_norm = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            value.append(1.0 / delta - 1.0 / s_norm)
    return np.array(value)


def rfzero(
    Fun: Callable[[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float], NDArray[np.floating]],
    x: float,
    itbnd: int,
    eigval: NDArray[np.floating],
    alpha: NDArray[np.floating],
    delta: float,
    tol: float = 1e-12,
) -> Tuple[float, float, int]:
    dx: float = abs(x) / 2 if x != 0 else 0.5
    a: float = x
    fa = Fun(np.array([a]), eigval, alpha, delta)[0]
    b: float = x + dx
    fb = Fun(np.array([b]), eigval, alpha, delta)[0]
    count = 2

    while np.sign(fa) == np.sign(fb):
        dx *= 2
        b = x + dx
        fb = Fun(np.array([b]), eigval, alpha, delta)[0]
        count += 1
        if count > itbnd:
            break

    c: float = b
    fc = fb
    for _ in range(itbnd):
        if fb == 0 or abs(b - c) < tol:
            break
        if np.sign(fb) == np.sign(fc):
            c, fc = a, fa
        m = 0.5 * (c - b)
        toler = 2.0 * tol * max(abs(b), 1.0)
        if abs(m) <= toler:
            break
        d: float = m
        a = b
        fa = fb
        b += d if abs(d) > toler else (toler if b <= c else -toler)
        fb = Fun(np.array([b]), eigval, alpha, delta)[0]
        count += 1
    return b, c, count


def trust(
    g: NDArray[np.floating],
    H: NDArray[np.floating],
    delta: float,
) -> Tuple[NDArray[np.floating], float, int, int, float]:
    tol: float = 1e-12
    H = np.array(H)
    eigval, V = eigh(H)
    mineig: float = float(np.min(eigval))
    jmin: int = int(np.argmin(eigval))
    alpha: NDArray[np.floating] = -V.T @ g
    sig: float = float(np.sign(alpha[jmin]) if alpha[jmin] != 0 else 1)

    if mineig > 0:
        coeff: NDArray[np.floating] = alpha / eigval
        s: NDArray[np.floating] = V @ coeff
        nrms: float = float(np.linalg.norm(s))
        if nrms <= 1.2 * delta:
            val: float = float(g.T @ s + 0.5 * s.T @ H @ s)
            return s, val, 1, 0, 0
        else:
            laminit: float = 0.0
    else:
        laminit = -mineig

    if seceqn(np.array([laminit]), eigval, alpha, delta)[0] > 0:
        b, _, count = rfzero(seceqn, laminit, 50, eigval, alpha, delta, tol)
        lambda_: float = b
    else:
        lambda_ = -mineig
        count = 0

    w: NDArray[np.floating] = eigval + lambda_
    coeff = np.where(w != 0, alpha / w, 0)
    s = V @ coeff
    nrms = float(np.linalg.norm(s))
    if nrms < 0.8 * delta:
        beta: float = float(np.sqrt(delta**2 - nrms**2))
        s += beta * sig * V[:, jmin]
    val = float(g.T @ s + 0.5 * s.T @ H @ s)
    posdef: int = 1 if mineig > 0 else 0
    return s, val, posdef, count, lambda_
