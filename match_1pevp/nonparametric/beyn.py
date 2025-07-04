from collections.abc import Callable
import numpy as np
from numpy.linalg import eigvals, svd
from match_1pevp.helpers.solve import solveLinearSystem as solveLS

def beyn(L : Callable[[complex], np.ndarray], center : float, radius : float, lhs : np.ndarray,
         rhs : np.ndarray, N_quad : int, rank_tol : float, hankel : int = 1
         ) -> np.ndarray[complex]:
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
    # sampling
    ts = center + radius * np.exp(1j * np.linspace(0., 2 * np.pi, N_quad + 1)[: -1])
    res_flat = np.array([(lhs @ solveLS(L(t), rhs)).reshape(-1) for t in ts])
    
    dft = ts.reshape(-1, 1) ** (1 + np.arange(2 * hankel))
    quad = dft.T * (ts - center)
    As = [A_flat.reshape(lhs.shape[0], rhs.shape[1]) for A_flat in quad @ res_flat]
    H0 = np.block([[As[i + j] for j in range(hankel)] for i in range(hankel)])
    H1 = np.block([[As[i + j + 1] for j in range(hankel)] for i in range(hankel)])

    u, s, vh = svd(H0)
    r_eff = np.where(s > rank_tol * s[0])[0][-1] + 1
    u, s, vh = u[:, : r_eff], s[: r_eff], vh[: r_eff, :]
    B = u.T.conj() @ H1 @ (vh.T.conj() / s[..., None, :])
    vals = eigvals(B)
    vals = vals[abs(vals - center) <= radius]
    return vals

def beyn_adapt(L : Callable[[complex], np.ndarray], center : float, radius : float,
               lhs : np.ndarray, rhs : np.ndarray, N_quad : int, rank_tol : float,
               hankel : int = 1, abs_tol_check : float = -1., dhankel_step : int = 1,
               dhankel_max : int = 5) -> np.ndarray[complex]:
    """    
    This function computes approximate eigenvalues of non-parametric eigenproblems through an adaptive version of Beyn's contour integral method
    
    Parameters:
    L: lambda function defining matrix in eigenproblem
    center: center of contour (disk)
    radius: radius of contour (disk)
    lhs: left-sketching matrix
    rhs: right-sketching matrix
    N_quad: number of quadrature points
    rank_tol: tolerance for rank truncation
    hankel: initial size of block-Hankel matrices
    abs_tol_check: tolerance for termination based on small Hankel spectral norm
    dhankel_step: step size for block-Hankel matrices size change
    dhankel_max: max number of explored block-Hankel matrices
    
    Returns:
    vals: approximate eigenvalues
    """
    N, M = lhs.shape[0], rhs.shape[1]
    # sampling
    ts = center + radius * np.exp(1j * np.linspace(0., 2 * np.pi, N_quad + 1)[: -1])
    res_flat = np.array([(lhs @ solveLS(L(t), rhs)).reshape(-1) for t in ts])

    H0 = np.empty((0, 0), dtype = complex)
    hankel_prev, As = 0, []
    for hankel_eff in range(hankel, hankel + dhankel_step * dhankel_max, dhankel_step):
        dft = ts.reshape(-1, 1) ** (1 + np.arange(2 * hankel_prev, 2 * hankel_eff))
        quad = dft.T * (ts - center)
        As_flat = quad @ res_flat
        As += [A_flat.reshape(N, M) for A_flat in As_flat]

        dhankel = hankel_eff - hankel_prev
        H0 = np.pad(H0, [(0, dhankel * N), (0, dhankel * M)])
        for i in range(hankel_eff):
            for j in range(hankel_prev, hankel_eff):
                H0[i * N : (i + 1) * N, j * M : (j + 1) * M] = As[i + j]
                if i < hankel_prev:
                    H0[j * N : (j + 1) * N, i * M : (i + 1) * M] = As[i + j]
        u, s, vh = svd(H0)
        if s[0] < abs_tol_check: return np.empty(0, dtype = complex)
        r_eff = np.where(s > rank_tol * s[0])[0][-1] + 1
        if r_eff < min(*H0.shape): break
        hankel_prev = hankel_eff
    else:
        return np.empty(0, dtype = complex)
    u, s, vh = u[:, : r_eff], s[: r_eff], vh[: r_eff, :]
    H1 = np.block([[As[i + j + 1] for j in range(hankel_eff)] for i in range(hankel_eff)])
    B = u.T.conj() @ H1 @ (vh.T.conj() / s[..., None, :])
    vals = eigvals(B)
    vals = vals[abs(vals - center) <= radius]
    return vals

