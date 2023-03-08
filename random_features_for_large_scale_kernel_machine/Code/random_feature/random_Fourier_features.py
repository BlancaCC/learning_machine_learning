"""
Compute Random Fourier Feature
Author: Blanca Cano 
Date: 03-08-23
"""
from typing import Callable
import numpy as np 
from random_samples import generate_random_examples

def random_fourier_features(
        fourier_transform:Callable[[np.array], float],
        input_dimension: int, 
        output_dimension: int
        ) -> Callable[[np.array], np.array]:
    '''
        Compute Random Fourier Features 

    Parameters
    --------------
    fourier_transform: Fourier transform of a positive definite shift invariant kernel K: R^d -> R
    input_dimension: d, input data dimension. 
    output_dimension: the dimension where the randomized feature map z goes 

    Return
    -------------
    z: a randomized feature map z(x): R^d -> R^D
    it verify 
    z(x) z(y).T \approx k(x-y)

    Example
    -------------
    >>> import numpy as np
    >>> from kernels import gaussian_fourier_transform
    >>> np.random.seed(1)
    >>> input_dimension = 4
    >>> output_dimension = 2
    >>> z = random_fourier_features(gaussian_fourier_transform, input_dimension, output_dimension)
    >>> x = np.array([1,2,1,4])
    >>> z(x)
    array([-0.87042363,  0.97120354])
    '''
    heuristic_k = 4/input_dimension
    W = generate_random_examples(fourier_transform, input_dimension, output_dimension, heuristic_k)
    D = np.random.uniform(0, 2*np.pi, output_dimension)

    z = lambda x: np.sqrt(2/output_dimension)*np.cos(
        np.sum(np.multiply(W, np.repeat([x], output_dimension, axis=0)), axis=1)
        +  
        D
       ) 
    return z
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()