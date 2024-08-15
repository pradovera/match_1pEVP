import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from match_1pevp import train, evaluate
from match_1pevp.nonparametric import beyn, loewner
from helpers_test import runTest

def define_problem(method : str):
    center, radius = -1., 1.
    N, l, k, f0, f1, t1, t2 = 5000, np.pi, .02, -.1, .05, 1., 2. # scalar constants appearing in the problem
    stiff = sparse.diags([-2 * np.ones(N - 1)] + [np.ones(N - 2)] * 2, [0, -1, 1], format = "csc") * (N / l) ** 2
    eye = sparse.eye(N - 1, format = "csc", dtype = complex) # identity
    # parametric matrix L(z,p) that defines the pEVP
    L_base = lambda z, p: k * stiff + (f0 - z - f1 * np.exp(-t1 * z) - p * np.exp(-t2 * z)) * eye
    L = lambda z, p: L_base(z + center, p) # center at z=0
    
    # define parameter range
    p_range = [-.1, .1]
    
    # define parameters for training
    l_sketch = 30 # number of sketching directions in Beyn's method
    lhs = np.random.randn(l_sketch, N - 1) + 1j * np.random.randn(l_sketch, N - 1) # left sketching matrix
    rhs = np.random.randn(N - 1, l_sketch) + 1j * np.random.randn(N - 1, l_sketch) # right sketching matrix
    
    if method == "beyn":
        train_nonpar = lambda L: beyn(L, 0., radius, lhs, rhs, 1000, 1e-10, 5)
    elif method == "loewner":
        lint = 0 + 3*radius * np.exp(1j * np.linspace(1.4 * np.pi/2, 2.6 * np.pi/2, l_sketch)) # left interpolation points
        rint = 0 + 3*radius * np.exp(1j * np.linspace(-1.4 * np.pi/2, 0.6 * np.pi/2, l_sketch)) # right interpolation points
        train_nonpar = lambda L: loewner(L, 0., radius, lhs, rhs, lint, rint, 1000, 1e-10)
    
    cutoff = lambda x: x[np.abs(x) <= radius]
    bounds = [[-2, 0], [-1, 1]]
    
    return p_range, L, train_nonpar, [], cutoff, center, bounds

if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger('match_1pevp').setLevel(logging.INFO)

    np.random.seed(42)

    method = "beyn"
    # method = "loewner" # uncomment to use loewner
    
    p_range, L, train_nonpar, train_nonpar_args, cutoff, center, bounds = define_problem(method)
    
    tol = 1e-6 # tolerance for outer adaptive loop
    interp_kind = "spline7" # interpolation strategy (degree-7 splines)
    patch_width = 11 # minimum width of interpolation patches in case of bifurcations
    
    # train
    model, ps_train = train(L, train_nonpar, train_nonpar_args, cutoff, interp_kind,
                            patch_width, p_range, tol)
    
    # test
    ps = np.linspace(*p_range, 500) # testing grid
    ps_coarse = ps[::10] # coarse testing grid
    getApprox = lambda p: evaluate(model, ps_train, p, interp_kind, patch_width, cutoff)
    def getExact(p): # reference solution
        return train_nonpar(lambda z: L(z, p), *train_nonpar_args)
    val_app, val_ref, error = runTest(ps, 10, getApprox, getExact) # run testing routine
    val_app, val_ref = val_app + center, val_ref + center # shift to original range
    
    # plot approximation and error
    plt.figure(figsize = (15, 5))
    plt.subplot(141)
    plt.plot(np.real(val_ref[:, 0]), ps_coarse, 'ro')
    plt.plot(np.real(val_app[:, 0]), ps, 'b:')
    plt.plot(np.real(val_ref[:, 1:]), ps_coarse, 'ro')
    plt.plot(np.real(val_app), ps, 'b:')
    plt.legend(['exact', 'approx'])
    plt.xlim(*bounds[0]), plt.ylim(*p_range)
    plt.xlabel("Re(lambda)"), plt.ylabel("p")
    plt.subplot(142)
    plt.plot(np.imag(val_ref[:, 0]), ps_coarse, 'ro')
    plt.plot(np.imag(val_app[:, 0]), ps, 'b:')
    plt.plot(np.imag(val_ref[:, 1:]), ps_coarse, 'ro')
    plt.plot(np.imag(val_app), ps, 'b:')
    plt.legend(['exact', 'approx'])
    plt.xlim(*bounds[1]), plt.ylim(*p_range)
    plt.xlabel("Im(lambda)"), plt.ylabel("p")
    plt.subplot(143)
    plt.plot([0] * len(ps_train), ps_train, 'bx')
    plt.ylim(*p_range)
    plt.ylabel("sample p-points")
    plt.subplot(144)
    plt.semilogx(error, ps_coarse)
    plt.semilogx([tol] * 2, p_range, 'k:')
    plt.ylim(*p_range)
    plt.xlabel("lambda error"), plt.ylabel("p")
    plt.tight_layout(), plt.show()
