import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import norm
from sysid_problem import Model, Cost, IdentificationProblem
from ode_tableau import RK4_explicit

# Let"s build an example where an ode is described by 2 states and just 1 input 
# in the following form:
def ode(x, u, p, t, h):
    dx      = np.zeros(2)
    dx[0]   = 0.8*u
    dx[1]   = 0.5*(x[0]-x[1])
    return dx

# Define some simulation parameters
h       = 0.1                       # time step
T       = 10                        # total sim. time
tvect   = np.arange(0, T, h)        # time vector
x0      = np.array([1, 0])          # initial condition
u       = np.zeros(len(tvect))      # input vector 
Xtrue   = np.zeros((len(tvect), 2))

u[10:20]    = 1          # set input and initial condition
u[30:60]    = -0.5
Xtrue[0, :] = x0

# Integrate the ode to generate the true state and plot them
for i in range(len(tvect)-1):
    Xtrue[i+1, :] = RK4_explicit.integrate(ode, Xtrue[i, :], u[i], np.zeros(1), tvect[i], h, False)

# plt.plot(tvect, Xtrue[:, 0])
# plt.plot(tvect, Xtrue[:, 1])
# plt.title("true trajectory")

# Generate measurements by adding noises and plot them
Xmeas = Xtrue + 0.05*np.random.randn(Xtrue.shape[0], Xtrue.shape[1])
Xmeas[0,:] = x0
Umeas = u + 0.02*np.random.randn(u.shape[0])
# plt.figure()
# plt.plot(tvect, Xmeas[:, 0])
# plt.plot(tvect, Xmeas[:, 1])
# plt.plot(tvect, Umeas)
# plt.title("measured states")

# Put data into a dataframe (in order to use the IdentificationProblem class)
df = pd.DataFrame()
df["x1"]      = Xmeas[:, 0]
df["x2"]      = Xmeas[:, 1]
df["u"]       = Umeas
df["minutes"] = tvect

# We build the model class based on the original "true" ode
class Mdl1(Model):
    def __init__(self, h):
        self.Nx = 2
        self.Nu = 1
        self.Np = 2
        self.h  = h
        
    def generate_x(self, df: pd.Series) -> np.ndarray:
        x    = np.zeros(self.Nx)
        x[0] = df.x1
        x[1] = df.x2
        return x

    def generate_u(self, df: pd.Series) -> np.ndarray:
        u    = np.zeros(self.Nu)
        u[0] = df.u
        return u    

    def f(self, x: np.ndarray, u: np.ndarray, p: np.ndarray, t: float, h: float) -> np.ndarray:        
        dx = np.zeros(self.Nx)
        dx[0] = p[0] * u[0]
        dx[1] = p[1] * (x[0] - x[1])
        return dx    

    def f_p(self, x: np.ndarray, u: np.ndarray, p: np.ndarray, t: float, h: float) -> np.ndarray:
        J = np.zeros((self.Nx, self.Np))
        J[0, 0] = u[0]
        J[1, 1] = x[0] - x[1]
        return J


# Define the identification problem
mdl     = Mdl1(h)
problem = IdentificationProblem(df, mdl)

p = problem.solve(True)
problem.simulate_parameters(p, 30)
print(p)