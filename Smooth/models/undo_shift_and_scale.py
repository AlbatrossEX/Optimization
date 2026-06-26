import numpy as np
from numpy.typing import NDArray


def undo_shift_and_scale(
    Y_hat: NDArray[np.floating],
    x: NDArray[np.floating],
    maxnorm: float,
) -> NDArray[np.floating]:
    return Y_hat * maxnorm + x
