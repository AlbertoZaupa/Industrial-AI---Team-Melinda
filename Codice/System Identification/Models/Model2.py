import numpy as np
import pandas as pd

from sysid_problem import Model
from data_importer import import_data 


def import_model_data(cell_number: int = 13) -> pd.DataFrame:

    if cell_number % 2 == 0:
        cell_other = cell_number - 1
    else:
        cell_other = cell_number + 1

    df_this  = import_data(cell_number, False, True)
    df_other = import_data(cell_other,  False, True)
    df       = df_this.merge(df_other, how = "inner", on = "date", suffixes=(None, "_other"))
    df["rock_temperature"] = 8
    return df


class Model(Model):


    def __init__(self, h):
        self.Nx = 3
        self.Nu = 6
        self.Np = 15
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
        u[3] = df.pump_status_other
        u[4] = df.outlet_fluid_temperature_other
        u[5] = df.ventilators_speed_percentage / 50

        return u
    

    def f(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.array:
        
        dx = np.zeros(self.Nx)

        Tin     = x[0]
        Tout    = x[1]
        Tcell   = x[2]
        pump    = u[0]
        Tsp     = u[1]
        Trock   = u[2]
        pump_o  = u[3]
        Tout_o  = u[4]

        if pump > 0:    # pump  on
            dx[0] = - (Tin-Tsp)/h
            dx[1] = p[0]*(Tout-Tcell) 
            dx[2] = p[1]*(Tcell-Tin) + p[2]*(Tcell-Tout)

        else:           # pump off
            dx[0] = p[3]*(Tin-Tcell)
            dx[1] = p[4]*(Tout-Tcell) 
            dx[2] = p[5]*(Tcell-Tin) + p[6]*(Tcell-Tout)

        if pump_o > 0:  # other pump is on
            dx[1] += p[7]*(Tout_o + p[8]) 

        dx[0] += p[9]  * (Tin   - Trock - p[10])**2
        dx[1] += p[11] * (Tout  - Trock) # - p[12])
        dx[2] += p[13] * (Tcell - Trock) # - p[14])

        return dx
    

    def f_p(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.ndarray:
        
        J = np.zeros((self.Nx, self.Np))

        Tin     = x[0]
        Tout    = x[1]
        Tcell   = x[2]
        pump    = u[0]
        Tsp     = u[1]
        Trock   = u[2]
        pump_o  = u[3]
        Tout_o  = u[4]

        if pump > 0:    # pump  on
            J[1, 0] = Tout - Tcell
            J[2, 1] = Tcell - Tin
            J[2, 2] = Tcell - Tout

        else:           # pump off
            J[0, 3] = Tin - Tcell
            J[1, 4] = Tout - Tcell
            J[2, 5] = Tcell - Tin
            J[2, 6] = Tcell - Tout

        if pump_o > 0:  # other pump is on
            J[1, 7] = Tout_o + p[8]
            J[1, 8] = p[7]

        J[0,  9] = (Tin - Trock - p[10])**2
        J[0, 10] = -2 * p[9] * (Tin - Trock - p[10])
        J[1, 11] = Tout  - Trock # - p[12]
        # J[1, 12] = -p[11]
        J[2, 13] = Tcell - Trock # - p[14]
        # J[2, 14] = -p[13]

        return J
    
    def simulate(self, df: pd.DataFrame, p: np.ndarray):
        X = np.zeros((len(df), self.Nx))
        X[0, :] = self.generate_x(df.iloc[0])
        for i in range(len(df)-1):
            u = self.generate_u(df.iloc[i])
            X[i+1, :] = self.f(X[i, :], u, p, 0, self.h)
        return X

