import numpy as np
from .helpers.match import match
from .helpers.cluster import findClusters, mergeClusters
from .evaluate import evaluate

def matchData(data, return_dist):
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
                data[j - 1] = np.pad(data[j - 1], [(0, Nj - N)], constant_values = np.inf)
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
                dists[j] = np.pad(dists[j], [(0, N - Nj)] * 2, constant_values = np.inf)
    if return_dist: return np.array(data), dists
    return np.array(data)

def clusterMatchedData(p, data, dists, d_thresh, min_patch_deltap):
    """
    This function clusters matched data points based on their distances. It then merges nearby clusters.

    Parameters:
    p (numpy array): The array of data points.
    data (list of numpy arrays): The matched data. Each element of the list is a numpy array of data points.
    dists (list of numpy arrays): The distances between matched points. Each element of the list is a numpy array of distances.
    d_thresh (float): The distance threshold for clustering. Points closer than this distance are considered part of the same cluster.
    min_patch_deltap (float): The minimum distance between clusters for them to be considered separate.

    Returns:
    list of numpy arrays: The merged clusters. Each element of the list is a numpy array of data points in a cluster.
    """
    clusters = []
    for d in dists:
        d_inf = np.isinf(np.diag(d))
        d[d_inf, d_inf] = 0.
        clusters += [findClusters(d, d_thresh)]
    # find effective clusters on each patch by merging nearby clusters
    try:
        deltaps = p[1:] - p[:-1]
    except TypeError:
        deltaps = np.array([p[j + 1] - p[j] for j in range(len(p) - 1)])
    return mergeClusters(clusters, deltaps, min_patch_deltap)

def train(L, solve_nonpar, solve_nonpar_args, cutoff, interp_kind, patch_width,
          ps_start, tol, max_iter=100, d_thresh=1e-1, min_patch_deltap=1e-2,
          verbose=False):
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
    model_out: The trained model
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
    
    if verbose: print("Initial sampling at {} point(s)".format(len(ps)))
    data = []
    for p in ps:
        if verbose: print(".", end = "", flush = True)
        Lp = lambda z: L(z, p)
        data += [solve_nonpar(Lp, *solve_nonpar_args)]
    if verbose: print()
    # train model
    model_out = matchData(data, d_thresh is not None)
    data = list(model_out[0])
    if d_thresh is not None:
        clusters = clusterMatchedData(ps, *model_out, d_thresh,
                                      min_patch_deltap)
        model_out = (model_out[0], clusters)
    if np.isinf(tol): return model_out, ps
    
    # Adaptive refinement
    for _ in range(int(max_iter)):
        # test model
        if verbose: print("Adaptive match iteration: test at {} point(s)".format(len(ps_next)))
        val_pre, val_ref = [], []
        for p in ps_next:
            if verbose: print(".", end = "", flush = True)
            val_pre += [evaluate(model_out, ps, p, interp_kind,
                                 patch_width, cutoff)]
            Lp = lambda z: L(z, p)
            val_ref += [solve_nonpar(Lp, *solve_nonpar_args)]
        if verbose: print()
        to_be_refined = []
        for j in range(len(ps_next)):
            # get prediction error
            Nref, Npre = len(val_ref[j]), len(val_pre[j])
            if Nref == 0 and Npre == 0:
                if verbose: print("\t no eigenvalues at {}".format(ps_next[j]))
            else:
                v_ref, v_pre = val_ref[j], val_pre[j]
                if Nref < Npre:
                    v_ref = np.pad(v_ref, [(0, Npre - Nref)], constant_values = np.inf)
                elif Nref > Npre:
                    v_pre = np.pad(v_pre, [(0, Nref - Npre)], constant_values = np.inf)
                p_opt, d_opt = match(v_ref, v_pre)
                d_opt_diag = d_opt[p_opt[0], p_opt[1]]
                d_opt_diag = d_opt_diag[np.logical_not(np.isinf(d_opt_diag))]
                d_tot = np.max(d_opt_diag) if len(d_opt_diag) else 0.
                if verbose: print("\t error at {} = {}".format(ps_next[j], d_tot))
            if d_tot > tol: to_be_refined += [j]
        # refine model
        ps_next_new, pre_next_new, dps_next_new = [], [], []
        if len(to_be_refined) == 0: break
        ps_new = list(ps)
        for n_added, j in enumerate(to_be_refined):
            ctr, pre, step = ps_next[j], pre_next[j], dps_next[j]
            for ishift, shift in enumerate([- step, step]):
                ctrn = ctr + shift
                if len(ps_next_new) == 0 or abs(ctrn - ps_next_new[-1]) > 1e-10: # no duplicate entries
                    ps_next_new += [ctrn]
                    pre_next_new += [pre + n_added + ishift]
                    dps_next_new += [.5 * step]
            idx_add = pre + n_added + 1
            ps_new = ps_new[: idx_add] + [ctr] + ps_new[idx_add :]
            data = data[: idx_add] + [val_ref[j]] + data[idx_add :]
        ps, ps_next = np.array(ps_new), ps_next_new
        pre_next, dps_next = pre_next_new, dps_next_new
        # train model
        model_out = matchData(data, d_thresh is not None)
        data = list(model_out[0])
        if d_thresh is not None:
            clusters = clusterMatchedData(ps, *model_out, d_thresh,
                                          min_patch_deltap)
            model_out = (model_out[0], clusters)
    else:
        if verbose: print("Max number of refinement iterations reached!")
    return model_out, ps

