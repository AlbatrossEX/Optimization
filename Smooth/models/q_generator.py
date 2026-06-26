import numpy as np
from numpy.typing import NDArray
from typing import Optional


def generate_random_Q(n: int, p: int) -> NDArray[np.floating]:
    """Generating a random Gaussian matrix Q."""
    return np.random.normal(0, 1 / p, size=(n, p))


def haar_orthog_matrix(dimension: int) -> NDArray[np.floating]:
    A: NDArray[np.floating] = np.random.randn(dimension, dimension)
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def generate_random_hashing_matrix(
    n: int,
    p: int,
    seed: Optional[int] = None,
) -> NDArray[np.floating]:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(p, n)) / np.sqrt(p)


def generate_subspace_matrix(
    jlm_type: int,
    n: int,
    p: int,
    hash_param: Optional[int] = None,
) -> NDArray[np.floating]:
    if jlm_type == 2:
        assert n == p, "Identity matrix can only be used when n == p"
        return np.eye(p)
    if jlm_type == 3:
        Q_full = haar_orthog_matrix(n)
        return np.sqrt(n / p) * Q_full[:, :p]
    if jlm_type == 4:
        Q_full = haar_orthog_matrix(n)
        return Q_full[:, :p]
    return generate_random_hashing_matrix(n, p, seed=hash_param).T
