import numpy as np
from    math    import pi, sqrt, exp


def gaussian(std: float, samples: int):
    x = np.linspace(0, samples-1, samples)
    y = np.zeros(samples)
    for i in range(samples):
        y[i] = 1/(std*sqrt(2*pi)) * exp( - 0.5 * (x[i]/std)**2 )
    return y