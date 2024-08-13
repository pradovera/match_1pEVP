from collections.abc import Callable
import numpy as np
from numpy.linalg import eigvals, svd
from match_1pevp.helpers.solve import solveLinearSystem as solveLS

def loewner(L : Callable[[complex], np.ndarray], center : float, radius : float, lhs : np.ndarray, rhs : np.ndarray,
            lint : np.ndarray[complex], rint : np.ndarray[complex], N_quad : int, rank_tol : float) -> np.ndarray[complex]:
    '''
    Parameters:
        L: lambda function defining matrix in eigenproblem
        center: center of contour (disk)
        radius: radius of contour (disk)
        lhs: left-sketching matrix
        rhs: right-sketching matrix
        lint: left interpolation points
        rint: right interpolation points
        N_quad: number of quadrature points
        rank_tol: tolerance for rank truncation

    Returns:
        vals : approximate eigenvalues
    '''
    ts = center + radius * np.exp(1j * np.linspace(0., 2 * np.pi, N_quad + 1)[: -1])
    QR = np.array([(solveLS(L(t), rhs)) for t in ts])
    QL = np.array([(solveLS(L(t).T.conj(), lhs.T.conj())).T.conj() for t in ts])
    dft_l = np.array([(1 / (lint[i] - ts)) for i in range(len(lint))]) 
    dft_r = np.array([(1 / (rint[i] - ts)) for i in range(len(rint))])
    quad_l = dft_l * (ts - center) # left weights
    quad_r = dft_r * (ts - center) # right weights
    cauchy = 1.0 / (lint.reshape((-1,1)) - rint)
    H_eval_l = np.array([quad_l[i,:] @ QL[:,i,:] for i in range(len(lint))]) 
    H_eval_r = np.array([quad_r[i,:] @ QR[:,:,i] for i in range(len(rint))]) 

    # Loewner matrices 
    Lo = cauchy * (H_eval_l @ rhs - lhs @ H_eval_r.T)  # see eq.18 in https://doi.org/10.1007/s10915-022-01800-3
    So = cauchy * (np.diag(lint) @ H_eval_l @ rhs - lhs @ H_eval_r.T @ np.diag(rint))

    u, s, vh = svd(Lo)
    r_eff = np.where(s > rank_tol * s[0])[0][-1] + 1
    u, s, vh = u[:, : r_eff], s[: r_eff], vh[: r_eff, :]
    B = np.diag(1/s) @ u.T.conj() @ So @ vh.T.conj() 
    vals = eigvals(B)
    vals = vals[abs(vals - center) <= radius]
            
    return vals