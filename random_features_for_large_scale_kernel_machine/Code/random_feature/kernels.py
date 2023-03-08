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
    The formula comes from the article: 
    author = {Rahimi, Ali and Recht, Benjamin},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {J. Platt and D. Koller and Y. Singer and S. Roweis},
    publisher = {Curran Associates, Inc.},
    title = {Random Features for Large-Scale Kernel Machines},
    url = {https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf},
    volume = {20},
    year = {2007}

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> Y= np.array([[1,2], [3, 4]])
    >>> kernel_matrix = gaussian_kernel(X,Y)
    >>> print(kernel_matrix)
    [[1.00000000e+00 1.83156389e-02]
     [1.83156389e-02 1.00000000e+00]
     [1.12535175e-07 1.83156389e-02]]
    """
    d = distance.cdist(x, y, metric='euclidean')
    return np.exp(-0.5 * (d)**2)


def gaussian_fourier_transform(w: np.array)-> float:
    ''' Compute the gaussian fourier transform
    Parameters
    ----------
    w: vector from R^d

    Returns
    -------
    Gausssian fourier transform of w

    Example
    -------
    >>> import numpy as np
    >>> w = np.array([1,2,3,4])
    >>> gaussian_fourier_transform(w)
    7.748596298045689e-09
    ''' 
    D = len(w)
    norm = w.dot(w)

    return (2*np.pi)**(-D/2)* np.exp(-norm/2)

if __name__ == "__main__":
    import doctest
    doctest.testmod()