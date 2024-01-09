import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x