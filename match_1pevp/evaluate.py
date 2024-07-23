import numpy as np
from .helpers.interp1d import interp1d_get_local_idx, interp1d_fast, interp1d_inf

def evaluate(model, ps, p, interp_kind, patch_width, cutoff):
    """
    This function evaluates the approximated model for the given parametric eigenproblem.

    Parameters:
    model: The trained model.
    ps: The final grid of sample p-points.
    p: The value of p at which the evaluation is requested.
    interp_kind: string label of p-interpolation type.
    patch_width: The width of the stencil for interpolation.
    cutoff: Callable that cuts off bad eigenvalues.

    Returns:
    The evaluated values.
    """
    S = len(ps)
    has_clusters = isinstance(model, tuple)
    j = interp1d_get_local_idx(p, ps, "previous")
    if j >= S - 1: j = S - 2
    j_patch_start, j_patch_end = 0, S
    j_patch_start_wide, j_patch_end_wide = 0, S
    if patch_width is not None:
        j_patch_start = max(0, j - (patch_width - 1) // 2) # width is patch_width + 1
        j_patch_end = min(S, j + (patch_width + 3) // 2) # width is patch_width + 1
        j_patch_start_wide = max(0, j - patch_width) # width is 2 * patch_width + 1
        j_patch_end_wide = min(S, j + patch_width + 1) # width is 2 * patch_width + 1
    ps_eff = ps[j_patch_start : j_patch_end]
    ps_eff_wide = ps[j_patch_start_wide : j_patch_end_wide]
    interp = interp1d_fast(p, ps_eff, interp_kind) # Initialize the interpolation
    if has_clusters: # If the model has clusters, get the data and the cluster
        data, cluster = model[0], model[1][j]
    else: # If the model doesn't have clusters, get the data and create a cluster
        data, cluster = model, [[j] for j in range(model.shape[1])]
    values = np.empty(data.shape[1], dtype = complex)

    for c in cluster:
        # find effective stencil by excluding inf values
        inf_near = np.any(np.isinf(data[j : j + 2, c]), axis = 1) # check inf left and right
        if inf_near[0] and inf_near[1]: # inf both left and right
            values[c] = np.inf
            continue
        c_indirect = False
        if inf_near[0] or inf_near[1]:
            # rely on extrapolation of previous or next model (the one that has finite values)
            cluster_try = min(S - 2, j + 1) if inf_near[0] else max(0, j - 1) # choose left or right
            if has_clusters:
                for c_eff in model[1][cluster_try]:
                    if np.all([c_ in c_eff for c_ in c]):
                       break
                else: # too complex! bifurcation is changing with migrations involved
                    values[c] = np.inf
                    continue
                # must check if other model has same cluster
                if not np.all([c_eff_ in c for c_eff_ in c_eff]):
                    c_indirect = True
                    c_, c = c, c_eff # store c for later use and overwrite with other cluster
            
        inf_on_patch = np.any(np.isinf(data[j_patch_start : j_patch_end, c])) # check inf over whole patch
        if len(c) == 1: # explicit form
            if inf_on_patch:
                values_ = interp1d_inf(p, ps_eff_wide, # use wide patch
                                       data[j_patch_start_wide : j_patch_end_wide, c[0]], interp_kind)
            else:
                values_ = interp(data[j_patch_start : j_patch_end, c[0]])
        else: # implicit form
            # get local implicit forms
            j_patch_start_ = j_patch_start_wide if inf_on_patch else j_patch_start
            j_patch_end_ = j_patch_end_wide if inf_on_patch else j_patch_end
            poly = np.empty((j_patch_end_ - j_patch_start_, len(c) + 1), dtype = complex)
            for k in range(j_patch_start_, j_patch_end_):
                poly[k - j_patch_start_] = np.poly(data[k, c])
            # interpolate implicit forms
            if inf_on_patch:
                poly_interpolated = interp1d_inf(p, ps_eff_wide, poly, interp_kind) # use wide patch
                poly_interpolated[0] = 1.
            else:
                poly_interpolated = interp(poly)
            # get implicitly defined eigenvalues
            try:
                values_ = np.roots(poly_interpolated)
            except np.linalg.LinAlgError:
                values_ = [np.inf] * len(c)
        if c_indirect: # values_ contains some extra unused values
            # look at support values on the finite side
            support_data = data[j + 1, c_] if inf_near[0] else data[j, c_]
            for k, val_ref in zip(c_, support_data):
                idx = np.argmin(np.abs(values_ - val_ref)) # closest predicted value
                values[k] = values_[idx]
        else:
            values[c] = values_
    return cutoff(values)
