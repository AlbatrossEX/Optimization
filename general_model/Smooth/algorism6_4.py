import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .models.build_vander import build_vander
from .models.maximize_lagrange_polynomial import maximize_lagrange_polynomial
from .models.shift_and_scale import shift_and_scale
from .models.undo_shift_and_scale import undo_shift_and_scale

def algorithm_6_4(
    Y: NDArray[np.floating],
    Delta: float,
    f: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:

    x_0: NDArray[np.floating] = Y[0, :].reshape(-1, 1)
    Y, scalefactor = shift_and_scale(Y)

    p_ini: int
    n: int
    p_ini, n = Y.shape
    p: int = int((n + 1) * (n + 2) / 2)

    X: NDArray[np.floating] = build_vander(Y)
    U: NDArray[np.floating] = np.eye(p)

    for i in range(p):
        ab_values: NDArray[np.floating] = np.abs(X @ U[i, :].reshape(-1, 1)).flatten()
        ab_values[:i] = 0
        value: float = float(np.max(ab_values))
        j_i: int = int(np.argmax(ab_values))

        if j_i < i and i < p_ini:
            j_i = i
            print("Point selection error in algorithm_6_4")

        if value > 0 and i < p_ini:
            Y[[i, j_i]] = Y[[j_i, i]]
            X[[i, j_i]] = X[[j_i, i]]
            f[[i, j_i]] = f[[j_i, i]]
        else:
            argmax: NDArray[np.floating]
            _max_value: float
            argmax, _max_value = maximize_lagrange_polynomial(
                U[i, :], np.zeros_like(x_0), Delta / scalefactor
            )
            new_point: NDArray[np.floating] = argmax.flatten().reshape(1, -1)
            Y = np.vstack([Y, new_point])
            f = np.vstack([f, [[np.inf]]])
            X = build_vander(Y)

        for j in range(i + 1, p):
            U[j, :] -= ((U[j, :] @ X[i, :]) / (U[i, :] @ X[i, :])) * U[i, :]

    Y = undo_shift_and_scale(Y, x_0.flatten(), scalefactor)
    return Y, f
