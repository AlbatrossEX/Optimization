import numpy as np
from numpy.typing import NDArray
from typing import List
from typing import Tuple

def bqmin(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    Low: NDArray[np.floating],
    Upp: NDArray[np.floating],
) -> List[NDArray[np.floating] | float]:
    """
    bqmin(A,B,Low,Upp) -> [X,f]
      Minimizes the quadratic 0.5 * X.T @ A @ X + B subject to Low<=X<=Upp using the
      projected gradient method with a (semi) exact line search.
      This will one day be replaced by a more efficient solver.
      This approach is not recommended for n>100.
     --INPUTS-----------------------------------------------------------------
     A       [dbl] [n-by-n] (Symmetric) Hessian matrix format
     B       [dbl] [n-by-1] Gradient vector
     Low       [dbl] [1-by-n] Vector of lower bounds assumed to be nonpositive
     Upp       [dbl] [1-by-n] Vector of upper bounds, must have Upp(j)>=0>=Low(j)
     --OUTPUTS----------------------------------------------------------------
     X       [dbl] [n-by-1] Approximate solution
     f       [dbl] Function value at X
    function [X,f] = bqmin(A,B,Low,Upp)
     --INTERMEDIATE-----------------------------------------------------------
     G       [dbl] [n-by-1]  Gradient at X
     it      [dbl] Iteration counter
     pap     [dbl] The A norm of the projected gradient
     Projg   [dbl] [n-by-1]  Projected gradient at X
     t       [dbl] Step length along projected gradient
    """
    # Internal Parameters
    n: int = np.shape(A)[1]
    maxit: int = 5000
    pgtol: float = 1e-13
    X: NDArray[np.floating] = np.zeros(n)
    f: float = float(X.T @ (0.5 * A @ X + B))
    G: NDArray[np.floating] = A @ X + B
    Projg: NDArray[np.floating] = X - np.maximum(np.minimum(X - G, Upp), Low)
    it: int = 0
    while it < maxit and np.linalg.norm(Projg, 2) > pgtol:
        it += 1
        t: float = 1.0
        pap: float = float(Projg.T @ (A @ Projg))
        if pap > 0:
            t = np.minimum(1, (Projg.T @ G) / pap)
        X = X - t * Projg
        f = float(X.T @ (0.5 * A @ X + B))
        G = A @ X + B
        Projg = X - np.maximum(np.minimum(X - G, Upp), Low)

    return [X, f]

def convertion () -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    sss