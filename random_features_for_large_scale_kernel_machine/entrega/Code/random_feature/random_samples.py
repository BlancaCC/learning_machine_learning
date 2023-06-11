"""
Generate Sample from P a density probability function
Author: Blanca Cano Camarero
Date: 03-08-23
"""
from typing import Callable 
import numpy as np

def generate_random_vector(p:Callable, dimension:int, k:float  = 2)-> np.array: 
    '''Generate a random vector from p a density probability function 
    Parameters
    ----------------
    p: a density probability function, p: R^dimension -> [0,1]
    dimension: the length of the generate vector
    k: the maximum in absolute value coefficient the generate vector may have
    ([-k,k] defines the domain of a uniform random variable) 
    Return 
    -----------
    w: vector from p a density probability function 
    Example
    ----------
    >>> from kernels import gaussian_fourier_transform
    >>> import numpy as np
    >>> dimension = 4
    >>> k = 3
    >>> np.random.seed(1)
    >>> generate_random_vector(gaussian_fourier_transform, dimension, k)
    array([ 0.02026566,  0.16296105, -1.27152247,  0.07199538])
    '''
    while True: 
        w = np.random.uniform(-k, k, dimension)
        u = np.random.uniform(0,1,1)
        if u[0] <= p(w):
            return w
            

def generate_random_examples(p:Callable, dimension:int, n_examples: int, k:float  = 2) -> np.ndarray:
    '''Generate a random vector from p a density probability function 

    Parameters
    ----------------
    p: a density probability function, p: R^dimension -> [0,1]
    dimension: the length of the generate vector
    k: the maximum in absolute value coefficient the generate vector may have
    ([-k,k] defines the domain of a uniform random variable) 

    Return 
    -----------
    w: vector from p a density probability function 

    Example
    ----------
    >>> from kernels import gaussian_fourier_transform
    >>> import numpy as np
    >>> dimension = 4
    >>> n_examples = 5
    >>> k = 3
    >>> np.random.seed(1)
    >>> W = generate_random_examples(gaussian_fourier_transform, dimension, n_examples, k/2)
    >>> W
    array([[ 0.46702057, -0.12688312,  1.03697172,  0.36090271],
           [ 0.10065235,  1.12209072,  0.31084176,  0.84494491],
           [-0.78107527,  0.97360619,  0.20851579, -0.47688404],
           [-0.49060391,  0.05414828,  0.53078049, -0.48671617],
           [ 0.00844402,  0.06790044, -0.52980103,  0.02999808]])
    '''
    W = np.array([
        generate_random_vector(p, dimension, 5/dimension)
        for _ in range(n_examples )
    ])
    return W

if __name__ == "__main__":
    import doctest
    doctest.testmod()