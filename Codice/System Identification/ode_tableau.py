"""
ODE solve module.

This module contains a class to solve ODEs using Runge-Kutta tableus; once the tableu is 
initialized, it can be used to integrate any function of the form 
    dx/dt = fun(x, u, t)
where x is the state, u is the input and t is the time.

At the end of the file it's also proposed an example of use, where a car model is integrated.

@author: Matteo Dalle Vedove (matteodv99tn@gmail.com)
"""


import numpy as np


#   ____ _                 _                 _                           _        _   _             
#  / ___| | __ _ ___ ___  (_)_ __ ___  _ __ | | ___ _ __ ___   ___ _ __ | |_ __ _| |_(_) ___  _ __  
# | |   | |/ _` / __/ __| | | '_ ` _ \| '_ \| |/ _ \ '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
# | |___| | (_| \__ \__ \ | | | | | | | |_) | |  __/ | | | | |  __/ | | | || (_| | |_| | (_) | | | |
#  \____|_|\__,_|___/___/ |_|_| |_| |_| .__/|_|\___|_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
#                                     |_|                                                           
class RKTableu:
    def __init__(self, 
                 a: np.ndarray, 
                 b: np.array, 
                 c: np.array):
        """ 
        Initializes a Runge-Kutta tableu.
        """

        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

        self.order       = self.b.size
        self.is_explicit = np.all(np.triu(a) == 0) # check if the tableu is explicit
    

    def integrate(self, 
                  fun:              callable, 
                  x0:               np.array, 
                  u:                np.array, 
                  p:                np.array,
                  t0:               float, 
                  h:                float,
                  compute_jacobian: bool = False,
                  fun_dp:           callable = None) -> np.array: 
        """
        Uses the tableu to integrate the function fun from t0 to t0+h given an input u.

        Args:
            fun (callable):             function to integrate.
            x0 (np.array):              initial state.
            u (np.array):               input of the function.
            p (np.array):               parameters of the model.
            t0 (float):                 initial integration time.
            h (float):                  integration step lenght.
            compute_jacobian (bool):    if True, the jacobian of fun w.r.t. the parameters p is 
                                        computed.
            fun_dp (callable):          function that computes the derivative of fun w.r.t. the 
                                        parameters p.

        Returns:    state at t0+h.
                    jacobian of the integration w.r.t. the parameters (optional).
        """
        
        if not self.is_explicit:
            raise NotImplementedError('Implicit Runge-Kutta tableus are not implemented yet')
        
        if compute_jacobian and (fun_dp is None):
            raise ValueError('If compute_jacobian is True, fun_dp must be provided')

        K       = np.zeros((x0.size, self.order))
        x_next  = x0
        J_p     = np.zeros((x0.size, p.size))

        for i in range(self.order):

            x = x0
            for j in range(i):
                x  += h * self.a[i][j] * K[:,j]   
            t       = t0 + self.c[i] * h
            K[:,i]  = fun(x, u, p, t, h)
            x_next += h * self.b[i] * K[:,i]

            if compute_jacobian:
                J_p += h * self.b[i] * fun_dp(x, u, p, t, h)
        
        if compute_jacobian:
            return x_next, J_p
        return x_next


#  _____     _     _                      
# |_   _|_ _| |__ | | ___  __ _ _   _ ___ 
#   | |/ _` | '_ \| |/ _ \/ _` | | | / __|
#   | | (_| | |_) | |  __/ (_| | |_| \__ \
#   |_|\__,_|_.__/|_|\___|\__,_|\__,_|___/         
                           
RK1_explicit_euler = RKTableu(
    a = [[0]],
    b = [1],
    c = [0])


RK2_Heun = RKTableu(
    a = [[0,    0],
         [1,    0]],
    b = [ 1/2,  1/2],
    c = [ 0,    1])


RK4_explicit = RKTableu(
    a = [[0,    0,      0,      0],
         [1/2,  0,      0,      0],
         [0,    1/2,    0,      0],
         [0,    0,      1,      0]],
    b = [ 1/6,  1/3,    1/3,    1/6],
    c = [ 0,    1/2,    1/2,    1])


#  _____                           _      
# | ____|_  ____ _ _ __ ___  _ __ | | ___ 
# |  _| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \
# | |___ >  < (_| | | | | | | |_) | |  __/
# |_____/_/\_\__,_|_| |_| |_| .__/|_|\___|
#                           |_|           

if __name__ == '__main__':
    """
    Example of use
    
    In this case we integrate a car model; the states are respectively position and velocity, while
    the input is the acceleration.
    """
    import matplotlib.pyplot as plt

    def car(x, u, t):       # car model dynamics
        dx    = np.zeros(2)
        dx[0] = x[1]        # position update
        dx[1] = u           # velocity update
        return dx

    x0 = np.array([0, 0])   # initial state
    h  = 0.01               # integration step
    TT = 1000               # number of integration steps

    t  = np.linspace(0, TT*h, TT)   # time vector
    u  = np.sin(t)          # sinusoidal input
    X  = np.zeros((2, TT))  # state matrix (2 states for TT time steps)

    for k in range(TT-1):   # integrate the dynamics
        X[:,k+1] = RK4_explicit.integrate(car, X[:,k], u[k], k*h, h)

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t, X[0,:])
    ax[0].legend(['Position'])
    ax[0].grid(linestyle='--', linewidth=0.5, color='lightgrey')
    ax[1].plot(t, X[1,:])
    ax[1].legend(['Velocity'])
    ax[1].grid(linestyle='--', linewidth=0.5, color='lightgrey')
    ax[2].plot(t, u)
    ax[2].legend(['Acceleration'])
    ax[2].grid(linestyle='--', linewidth=0.5, color='lightgrey')
    plt.show()
