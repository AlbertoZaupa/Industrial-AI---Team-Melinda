import numpy as np

from math import exp

def logistic_function(x: float, pars: np.ndarray) -> float:
    """
    Parametric logistic function
    
    Parameters
    ----------
    x : float
        Independent variable.
    pars : np.ndarray (of size 3)
        Parameters of the logistic function.

    Returns
    -------
    float
        Value of the logistic function.
    """

    L   = pars[0]
    k   = pars[1]
    x0  = pars[2]

    try:
        y   = L / (1 + exp(-k * (x - x0)))
    except OverflowError:
        y   = float("inf")
    return y


def logistic_function_jacobian(x: float, pars: np.ndarray) -> np.ndarray:
    """
    Jacobian of the parametric logistic function with respect to the parameters.

    Parameters
    ----------
    x : float
        Independent variable.
    pars : np.ndarray of size 3
        Parameters of the logistic function.

    Returns
    -------
    np.ndarray (of size 3)
        Jacobian of the logistic function with respect to the parameters L, k, x0.
    """
    
    L   = pars[0]
    k   = pars[1]
    x0  = pars[2]

    e   = exp(-k * (x - x0))
    den = 1 + e
    j0  = 1 / den
    j1  = e * L * (x - x0) / (den**2)
    j2  = -e * L * k / (den**2)

    return np.array([j0, j1, j2])


def logistic_function_jacobian_state(x: float, pars: np.ndarray) -> float:
    """
    Jacobian of the parametric logistic function with respect to the input x.

    Parameters
    ----------
    x : float
        Independent variable.
    pars : np.ndarray of size 3
        Parameters of the logistic function.

    Returns
    -------
    float
        Jacobian of the logistic function with respect to the state variable x.
    """
    
    L   = pars[0]
    k   = pars[1]
    x0  = pars[2]

    e   = exp(-k * (x - x0))
    num = e * L * k
    den = (1 + e)**2

    return num / den
