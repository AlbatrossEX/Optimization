import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .build_vander import build_vander
from .polynomial_to_matrix import polynomial_to_matrix
from .trust import trust


def maximize_lagrange_polynomial(
    L: NDArray[np.floating],
    x_0: NDArray[np.floating],
    Delta: float,
) -> Tuple[NDArray[np.floating], float]:
    max_val: float = 0.0
    argmax: NDArray[np.floating] = x_0
    Q: NDArray[np.floating]
    c: NDArray[np.floating]
    _b: float
    Q, c, _b = polynomial_to_matrix(L, x_0.shape[0])
    for j in [0, 1]:
        linear_term: NDArray[np.floating] = Q @ x_0 + c.reshape(-1, 1)
        Q = (-1) ** j * Q
        linear_term = (-1) ** j * linear_term
        if not np.any(Q):
            test_step: NDArray[np.floating] = -linear_term / np.linalg.norm(linear_term)
            test_step = test_step * Delta
        else:
            test_step, *_ = trust(linear_term.flatten(), Q, Delta)
        test_x: NDArray[np.floating] = x_0 + test_step.reshape(-1, 1)
        test_val: float = float(np.abs(L @ build_vander(test_x.T).T)[0])
        if test_val > max_val:
            max_val = test_val
            argmax = test_x
    return argmax, max_val
