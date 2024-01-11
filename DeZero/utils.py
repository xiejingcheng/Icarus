import numpy as np

def asArray(x):
    if np.isscalar(x):
        return np.array(x)
    return x