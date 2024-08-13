from collections.abc import Callable
import numpy as np
from numpy.linalg import eigvals, svd
from match_1pevp.helpers.solve import solveLinearSystem as solveLS

def beyn(L : Callable[[complex], np.ndarray], center : float, radius : float, lhs : np.ndarray,
         rhs : np.ndarray, N_quad : int, rank_tol : float, hankel : int = 1) -> np.ndarray[complex]:
    """    
    This function computes approximate eigenvalues of non-parametric eigenproblems through Beyn's contour integral method
    
    Parameters:
    L: lambda function defining matrix in eigenproblem
    center: center of contour (disk)
    radius: radius of contour (disk)
    lhs: left-sketching matrix
    rhs: right-sketching matrix
    N_quad: number of quadrature points
    rank_tol: tolerance for rank truncation
    hankel: size of block-Hankel matrices
    
    Returns:
    vals: approximate eigenvalues
    """
    ts = center + radius * np.exp(1j * np.linspace(0., 2 * np.pi, N_quad + 1)[: -1])
    res_flat = np.array([(lhs @ solveLS(L(t), rhs)).reshape(-1) for t in ts])
    dft = ts.reshape(-1, 1) ** (1 + np.arange(2 * hankel))
    quad = dft.T * (ts - center)
    As_flat = quad @ res_flat
    As = As_flat.reshape(2 * hankel, lhs.shape[0], rhs.shape[1])
    H0 = np.block([[As[i + j] for j in range(hankel)] for i in range(hankel)])
    H1 = np.block([[As[i + j + 1] for j in range(hankel)] for i in range(hankel)])
    u, s, vh = svd(H0)
    r_eff = np.where(s > rank_tol * s[0])[0][-1] + 1
    u, s, vh = u[:, : r_eff], s[: r_eff], vh[: r_eff, :]
    B = u.T.conj() @ H1 @ (vh.T.conj() / s[..., None, :])
    vals = eigvals(B)
    vals = vals[abs(vals - center) <= radius]
    return vals
