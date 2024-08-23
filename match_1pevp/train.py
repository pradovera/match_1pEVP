from typing import Optional, Union
from collections.abc import Callable
import numpy as np
from .helpers.match import match
from .helpers.cluster import findClusters, mergeClusters
from .evaluate import evaluate
from .helpers import logger as log

def matchData(data : list[np.ndarray], return_dist : bool) -> Union[list[np.ndarray, list[np.ndarray]], np.ndarray]:
    """
    This function matches data, even if it's unbalanced. It matches the data with infinity where necessary to make it balanced.

    Parameters:
    data (list of numpy arrays): The data to be matched. Each element of the list is a numpy array of data points.
    return_dist (bool): If True, the function also returns the distances between matched points.

    Returns:
    numpy array: The matched data. Each element of the array corresponds to the matched data points.
    list (optional): The distances between matched points. Only returned if return_dist is True.
    """
    # If return_dist is True, initialize an empty list to store distances
    if return_dist: dists = []
    
    # Enumerate over the data
    for j, dataj in enumerate(data):
        dataj = data[j]
        Nj = len(dataj)
        
        # If this is the first element, store its length in N
        if j == 0:
            N = Nj
        else: # perform matching
            if Nj < N: # If there are too few new values, add inf values
                dataj = np.pad(dataj, [(0, N - Nj)], constant_values = np.inf)
            elif len(dataj) > N: # If there are too many new values, match with previous
                data[j - 1] = np.pad(data[j - 1], [(0, Nj - N)],
                                     constant_values = np.inf)
                N = Nj
            # now the problem of matching data[j - 1] and dataj is balanced
            p_opt, d_opt = match(data[j - 1], dataj)
            data[j] = dataj[p_opt[1]]
            # If return_dist is True, store the distances
            if return_dist: dists += [d_opt[:, p_opt[1]]]
    for j, dataj in enumerate(data): # add missing entries in unbalanced case
        Nj = len(dataj)
        if Nj < N:
            data[j] = np.pad(data[j], [(0, N - Nj)], constant_values = np.inf)
            if j < len(data) - 1 and return_dist:
                dists[j] = np.pad(dists[j], [(0, N - Nj)] * 2,
                                  constant_values = np.inf)
    if return_dist: return np.array(data), dists
    return np.array(data)

def postprocessModel(ps : np.ndarray[float], data : list[np.ndarray],
                     d_thresh : Optional[float], min_patch_deltap : float
                     ) -> tuple[np.ndarray, Optional[list[list[list[int]]]]]:
    """
    This function reorders and matches the computed spectrum, clustering and merging matched data points based on their distances.

    Parameters:
    ps (numpy array): The array of sample points.
    data (list of numpy arrays): List of computed spectra to be matched.
    d_thresh (float): The distance threshold for clustering. Points closer than this distance are considered part of the same cluster.
    min_patch_deltap (float): The minimum distance between clusters for them to be considered separate.

    Returns:
    data: Sorted data
    clusters: Computed clusters, if required
    """
    if d_thresh is None:
        data = matchData(data)
        clusters = None
    else:
        data, dists = matchData(data, True)
        clusters = []
        for d in dists:
            d_inf = np.isinf(np.diag(d))
            d[d_inf, d_inf] = 0.
            clusters += [findClusters(d, d_thresh)]
        # find effective clusters on each patch by merging nearby clusters
        deltaps = ps[1:] - ps[:-1]
        clusters = mergeClusters(clusters, deltaps, min_patch_deltap)
    return data, clusters

def computeMatchError(p : float, v_ref : np.ndarray, v_pre : np.ndarray) -> float:
    """
    This function computes the spectrum approximation error by a match step.

    Parameters:
    p (float): Current sample point.
    v_ref (numpy array): Exact spectrum.
    v_pre (numpy array): Approximate spectrum.

    Returns:
    d_tot: Approximation error
    """
    Nref, Npre = len(v_ref), len(v_pre)
    log.debug("Comparison at p={} between exact={} and approx={}".format(p, v_ref,
                                                                         v_pre))
    if Nref == 0 and Npre == 0:
        log.info("\t no eigenvalues at {}".format(p))
        return 0
    if Nref < Npre:
        v_ref = np.pad(v_ref, [(0, Npre - Nref)], constant_values = np.inf)
    elif Nref > Npre:
        v_pre = np.pad(v_pre, [(0, Nref - Npre)], constant_values = np.inf)
    p_opt, d_opt = match(v_ref, v_pre)
    d_opt_diag = d_opt[p_opt[0], p_opt[1]]
    log.debug("Reordered spectra are exact={} and approx={}, with errors={}".format(
                                     v_ref[p_opt[0]], v_pre[p_opt[1]], d_opt_diag))
    d_opt_diag = d_opt_diag[np.logical_not(np.isinf(d_opt_diag))]
    d_tot = np.max(d_opt_diag) if len(d_opt_diag) else 0.
    log.info("\t error at {} = {}".format(p, d_tot))
    return d_tot

def refineGrid(ps : np.ndarray[float], data : list[np.ndarray], ps_next_bad : list[float],
               pre_next_bad : list[int], dps_next_bad : list[float], data_bad : list[np.ndarray]
               ) -> tuple[np.ndarray[float], list[float], list[int], list[float], np.ndarray]:
    """
    Perform operation to refine grid based.

    Parameters:
    ps (numpy array): The array of sample points.
    data (list of numpy arrays): List of computed spectra.
    ps_next_bad (list of floats): List of sample points to be appended.
    pre_next_bad (list of ints): List of location indices for future refinements.
    dps_next_bad (list of floats): List of mesh-sizes for future refinements.
    data_bad (list of numpy arrays): List of spectra to be appended.

    Returns:
    ps: Array of sample points
    ps_next: Array of test points to be explored
    pre_next: List of location indices for future refinements
    dps_next: List of mesh-sizes for future refinements
    data: Updated list of computed spectra.
    """
    ps_next, pre_next, dps_next = [], [], []
    ps, data = list(ps), list(data)
    for i, (p, pre, step, datum) in enumerate(zip(ps_next_bad, pre_next_bad,
                                                  dps_next_bad, data_bad)):
        for ishift, shift in enumerate([- step, step]):
            ps_next += [p + shift]
            pre_next += [pre + i + ishift]
            dps_next += [.5 * step]
        idx_add = pre + i + 1
        ps = ps[: idx_add] + [p] + ps[idx_add :]
        data = data[: idx_add] + [datum] + data[idx_add :]
    return np.array(ps), ps_next, pre_next, dps_next, data

def train(L : Callable[[complex, float], np.ndarray],
          solve_nonpar : Callable[[Callable[[complex], np.ndarray], ...], np.ndarray], # type: ignore
          solve_nonpar_args : list[...], # type: ignore
          cutoff : Callable[[np.ndarray[complex]], np.ndarray[complex]], interp_kind : str,
          patch_width : int, ps_start : Union[np.ndarray[float], list[float]], tol : float,
          max_iter : int = 100, d_thresh : Optional[float] = 1e-1, min_patch_deltap : float = 1e-2
          ) -> tuple[tuple[np.ndarray[complex], Optional[list[list[list[int]]]]], np.ndarray[float]]:
    """
    This function trains the approximation model with the data taken from the parametric eigenproblem.
    
    Parameters
    L: lambda function defining matrix in eigenproblem
    solve_nonpar: lambda function defining non-parametric eigensolver
    solve_nonpar_args: arguments of solve_nonpar
    cutoff: callable that cuts off bad eigenvalues
    interp_kind: string label of p-interpolation type
    patch_width: width of stencil for interpolation
    ps_start: initial grid of sample p-points
    tol: tolerance epsilon for adaptivity
    max_iter: maximum number of adaptivity iterations
    d_thresh: tolerance delta for bifurcation
    min_patch_deltap: width of stencil for implicit bifurcation management
    verbose: whether to track progress in console
    
    Returns:
    model_out: The trained model (tuple with eigenvalues and clusters)
    ps: The final grid of sample p-points
    """
    
    # initial sampling grid for the parameters-points
    ps = np.array(ps_start)
    dps = ps[1:] - ps[:-1]
    if any(abs(dps - dps[0]) > 1e-10):
        raise Exception("ps_start must contain equispaced points")
    ps_next = list(.5 * (ps[:-1] + ps[1:]))
    pre_next = list(range(len(ps_next)))
    dps_next = [.25 * dps[0]] * len(ps_next)
    
    log.info("Initial sampling at {} point(s)".format(len(ps)))
    spectra_list = []
    for p in ps:
        Lp = lambda z: L(z, p)
        spectra_list += [solve_nonpar(Lp, *solve_nonpar_args)]
        log.debug("Spectrum at p={} is {}".format(p, spectra_list[-1]))
    # train model
    model_out = postprocessModel(ps, spectra_list, d_thresh, min_patch_deltap)
    if np.isinf(tol): return model_out, ps
    
    # Adaptive refinement
    for _ in range(int(max_iter)):
        # test model
        log.info("Adaptive match iteration: test at {} point(s)".format(len(ps_next)))
        val_pre, val_ref = [], []
        for p in ps_next:
            val_pre += [evaluate(model_out, ps, p, interp_kind,
                                 patch_width, cutoff)]
            Lp = lambda z: L(z, p)
            val_ref += [solve_nonpar(Lp, *solve_nonpar_args)]
            log.debug("Spectrum at p={} is {}".format(p, val_ref[-1]))
        # get prediction error
        ps_next_bad, pre_next_bad, dps_next_bad, spectra_bad = [], [], [], []
        for j, (p, v_ref, v_pre) in enumerate(zip(ps_next, val_ref, val_pre)):
            d_tot = computeMatchError(p, v_ref, v_pre)
            if d_tot > tol:
                ps_next_bad += [p]
                pre_next_bad += [pre_next[j]]
                dps_next_bad += [dps_next[j]]
                spectra_bad += [v_ref]
        # refine model
        if len(ps_next_bad) == 0: break
        ps, ps_next, pre_next, dps_next, spectra_list = refineGrid(ps, model_out[0],
                                                          ps_next_bad, pre_next_bad,
                                                          dps_next_bad, spectra_bad)
        # train model
        model_out = postprocessModel(ps, spectra_list, d_thresh, min_patch_deltap)
    else:
        log.warning("Max number of refinement iterations reached!")
    return model_out, ps
