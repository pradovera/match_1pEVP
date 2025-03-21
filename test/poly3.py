import numpy as np
from matplotlib import pyplot as plt
from match_1pevp import train, evaluate
from match_1pevp.nonparametric import beyn, loewner
from helpers_test import runTest

def define_problem(method : str):
    radius = 4.
    A = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 0]])
    B = np.array([[0, 0, -2], [0, 0, -1], [0, 0, 0]])
    C = np.eye(3)
    # parametric matrix L(z,p) that defines the pEVP
    L = lambda z, p: A + p * B - z * C
    
    # define parameter range
    p_range = [-50., 50.]
    
    # define parameters for training
    l_sketch = 5 # number of sketching directions in Beyn's method
    lhs = np.random.randn(l_sketch, 3) + 1j * np.random.randn(l_sketch, 3) # left sketching matrix
    rhs = np.random.randn(3, l_sketch) + 1j * np.random.randn(3, l_sketch) # right sketching matrix
    
    if method == "beyn":
        train_nonpar = lambda L: beyn(L, 0., radius, lhs, rhs, 25, 1e-10, 1)
    elif method == "loewner":
        lint = 0 + 3*radius * np.exp(1j * np.linspace(1.4 * np.pi/2, 2.6 * np.pi/2, l_sketch)) # left interpolation points
        rint = 0 + 3*radius * np.exp(1j * np.linspace(-1.4 * np.pi/2, 0.6 * np.pi/2, l_sketch)) # right interpolation points
        train_nonpar = lambda L, radius: loewner(L, 0., radius, lhs, rhs, lint, rint, 30, 1e-10)
    
    cutoff = lambda x: x[np.abs(x) <= radius]
    bounds = [[-4, 4], [-4, 4]]
    
    return p_range, L, train_nonpar, [], cutoff, 0., bounds

if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger('match_1pevp').setLevel(logging.INFO)

    np.random.seed(42)

    method = "beyn"
    # method = "loewner" # uncomment to use loewner
    
    p_range, L, train_nonpar, train_nonpar_args, cutoff, center, bounds = define_problem(method)
    
    tol = 1e-2 # tolerance for outer adaptive loop
    interp_kind = "linear" # interpolation strategy (piecewise-linear hat functions)
    min_patch_deltap = 5 # minimum width of interpolation patches in case of bifurcations
    
    # train
    model, ps_train = train(L, train_nonpar, train_nonpar_args, cutoff, interp_kind,
                            None, p_range, tol, min_patch_deltap = min_patch_deltap)
    
    # test
    ps = np.linspace(*p_range, 1500) # testing grid
    getApprox = lambda p: evaluate(model, ps_train, p, interp_kind, None, cutoff)
    def getExact(p): # exact solution
        alpha = lambda p: (((3*(4*p**3+84*p**2-60*p-5+0j))**.5-18*p+9)/18) ** (1./3)
        beta = lambda p: (p-2)/3/alpha(p)
        versor1, versor2 = np.exp(1j*np.pi/3), np.exp(-1j*np.pi/3)
        val1 = lambda p: alpha(p)-beta(p)
        val2 = lambda p: -versor2*alpha(p)+versor1*beta(p)
        val3 = lambda p: -versor1*alpha(p)+versor2*beta(p)
        v_ref = np.array([val1(p), val2(p), val3(p)])
        return cutoff(v_ref)
    val_app, val_ref, error = runTest(ps, 1, getApprox, getExact) # run testing routine
    val_app, val_ref = val_app + center, val_ref + center # shift to original range
    
    # plot approximation and error
    plt.figure(figsize = (15, 5))
    plt.subplot(141)
    plt.plot(np.real(val_ref[:, 0]), ps, 'ro')
    plt.plot(np.real(val_app[:, 0]), ps, 'b:')
    plt.plot(np.real(val_ref[:, 1:]), ps, 'ro')
    plt.plot(np.real(val_app), ps, 'b:')
    plt.legend(['exact', 'approx'])
    plt.xlim(*bounds[0]), plt.ylim(*p_range)
    plt.xlabel("Re(lambda)"), plt.ylabel("p")
    plt.subplot(142)
    plt.plot(np.imag(val_ref[:, 0]), ps, 'ro')
    plt.plot(np.imag(val_app[:, 0]), ps, 'b:')
    plt.plot(np.imag(val_ref[:, 1:]), ps, 'ro')
    plt.plot(np.imag(val_app), ps, 'b:')
    plt.legend(['exact', 'approx'])
    plt.xlim(*bounds[1]), plt.ylim(*p_range)
    plt.xlabel("Im(lambda)"), plt.ylabel("p")
    plt.subplot(143)
    plt.plot([0] * len(ps_train), ps_train, 'bx')
    plt.ylim(*p_range)
    plt.ylabel("sample p-points")
    plt.subplot(144)
    plt.semilogx(error, ps)
    plt.semilogx([tol] * 2, p_range, 'k:')
    plt.ylim(*p_range)
    plt.xlabel("lambda error"), plt.ylabel("p")
    plt.tight_layout(), plt.show()
