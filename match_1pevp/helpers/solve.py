from typing import Union
from numpy import ndarray
from numpy.linalg import solve
from scipy.sparse import spmatrix
from scipy.sparse.linalg import factorized

def solveLinearSystem(A : Union[ndarray, spmatrix], b : ndarray) -> ndarray:
    if isinstance(A, ndarray): return solve(A, b)
    return factorized(A)(b)
