from numpy import ndarray
from numpy.linalg import solve
from scipy.sparse.linalg import factorized

def solveLinearSystem(A, b):
    if isinstance(A, ndarray): return solve(A, b)
    return factorized(A)(b)
