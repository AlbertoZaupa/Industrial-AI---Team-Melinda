import numpy as np
import pandas as pd

from sysid_problem import Model
from data_importer import import_data


def import_model_data(cell_number: int = 13) -> pd.DataFrame:
    return import_data(cell_number, False, True)

    
class Model(Model):


    def __init__(self, h):
        self.Nx = 3
        self.Nu = 3
        self.Np = 6
        self.h  = h


    def generate_x(self, df: pd.Series) -> np.array:
        
        x    = np.zeros(self.Nx)

        x[0] = df.inlet_fluid_temperature
        x[1] = df.outlet_fluid_temperature
        x[2] = df.cell_temperature

        return x


    def generate_u(self, df: pd.Series) -> np.array:
        
        u    = np.zeros(self.Nu)

        u[0] = df.pump_status
        u[1] = df.inlet_setpoint_temperature
        u[2] = df.rock_temperature

        return u
    

    def f(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.array:
        
        dx = np.zeros(self.Nx)

        Tin     = x[0]
        Tout    = x[1]
        Tcell   = x[2]
        pump    = u[0]
        Tsp     = u[1]
        Trock   = u[2]

        dx[0] = p[0]*(Tin-Tcell) - (Tin-Tsp)*pump/h + p[5]
        dx[1] = p[1]*(Tout-Tcell) 
        dx[2] = p[2]*(Tcell-Trock) + p[3]*(Tcell-Tin) + p[4]*(Tcell-Tout)

        return dx
    

    def f_p(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.ndarray:
        
        J = np.zeros((self.Nx, self.Np))

        Tin     = x[0]
        Tout    = x[1]
        Tcell   = x[2]
        pump    = u[0]
        Tsp     = u[1]
        Trock   = u[2]

        J[0, 0] = Tin - Tcell
        J[1, 1] = Tout - Tcell
        J[2, 2] = Tcell - Trock
        J[2, 3] = Tcell - Tin
        J[2, 4] = Tcell - Tout
        J[0, 5] = 1

        return J
    
    def simulate(self, df: pd.DataFrame, p: np.ndarray):
        X = np.zeros((len(df), self.Nx))
        X[0, :] = self.generate_x(df.iloc[0])
        for i in range(len(df)-1):
            u = self.generate_u(df.iloc[i])
            X[i+1, :] = self.f(X[i, :], u, p, 0, self.h)
        return X

