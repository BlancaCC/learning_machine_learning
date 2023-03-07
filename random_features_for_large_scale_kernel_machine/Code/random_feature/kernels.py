# Kernels 

# We need to implements
# Positive definite shift- invariant kernels

# ¨Here in wikipedia there are many kernels:
# url: https://en.wikipedia.org/wiki/Positive-definite_kernel
# Date: 03-07-23
from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance
from scipy import linalg


def gaussian_kernel(x:np.array,y:np.array) -> float: 
    """
    Parameters
    ----------
    x: real vector from R^d
    y: real vector from R^d
    sigma: 
    Returns
    -------
    Gaussian kernel: 

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2
    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(x, y, metric='euclidean')
    return np.exp(-0.5 * (d)**2)


def gaussian_fourier_tranform(w: np.array)-> float:
    '''
    W array 
    ''' 
    D = len(w)
    norm = w.dot(w)

    return (2*np.pi)**(-D/2)* np.exp(norm/2)


