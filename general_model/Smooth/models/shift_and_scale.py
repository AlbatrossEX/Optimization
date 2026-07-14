import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def shift_and_scale(Y: NDArray[np.floating]) -> Tuple[NDArray[np.floating], float]:
    Y_hat: NDArray[np.floating] = Y - Y[0, :]
    if Y_hat.shape[0] > 1:
        maxnorm: float = float(max(np.linalg.norm(row) for row in Y_hat))
        Y_hat = Y_hat / maxnorm
    else:
        maxnorm = 1.0
    return Y_hat, maxnorm
