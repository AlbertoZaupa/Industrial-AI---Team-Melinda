""" Overall system model.

This module embeds all testing performed on the different states independently.

Info
----
Author: Matteo Dalle Vedove (matteodv99tn@gmail.com)
Date:   13 February 2021
"""

import numpy as np
import pandas as pd

from sysid_problem import Model
from data_importer import import_data 
from sysid_functions import *

STATE_VARIABLES = [
    "inlet_fluid_temperature",
    "outlet_fluid_temperature",
    "cell_temperature"
    ]
INPUT_VARIABLES = [
    "pump_status", 
    "inlet_setpoint_temperature",
    "rock_temperature", 
    "time_free_run",
    "pump_status_other",
    "outlet_fluid_temperature_other",
    "time_free_run_other"
    ]


def import_model_data(cell_number: int = 13) -> pd.DataFrame:
    """ Given a cell number, it imports the useful data from the corresponding .csv file.

    Parameters
    ----------
    cell_number : int, optional
        Number of the cell to import data from; defaults to 13.

    Returns
    -------
    Pandas dataframe containing the useful data for the model.
    """

    if cell_number % 2 == 0:
        cell_other = cell_number - 1
    else:
        cell_other = cell_number + 1

    df_this                 = import_data(cell_number, False, True)
    df_other                = import_data(cell_other,  False, True)
    df                      = df_this.merge(df_other, how = "inner", on = "date", suffixes=(None, "_other"))
    rock_labels             = ["rock1_temperature", "rock2_temperature", "rock3_temperature"]
    df["rock_temperature"]  = df[rock_labels].mean(axis = 1)
    df["time_free_run"]     = 0
    df["time_free_run_other"] = 0

    for i in range(1, len(df)):
        if df.pump_status[i] == 1:
            df.time_free_run[i] = 0
        else:
            df.time_free_run[i] = df.time_free_run[i-1] + 1

        if df.pump_status_other[i] == 1:
            df.time_free_run_other[i] = 0
        else:
            df.time_free_run_other[i] = df.time_free_run_other[i-1] + 1

    return df


class Model(Model):


    def __init__(self, h):
        self.Nx         = len(STATE_VARIABLES)
        self.Nu         = len(INPUT_VARIABLES)
        self.Np         = 25
        self.h          = h
        self.t_delay    = 15
        self.name       = "Model1"


    def generate_x(self, df: pd.Series) -> np.array:
        x    = np.zeros(self.Nx)
        for i in range(self.Nx):
            x[i] = df[STATE_VARIABLES[i]]
        return x


    def generate_u(self, df: pd.Series) -> np.array:
        u    = np.zeros(self.Nu)
        for i in range(self.Nu):
            u[i] = df[INPUT_VARIABLES[i]]
        return u
    

    def f(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.array:
        dx = np.zeros(self.Nx)

        Tin         = x[0]
        Tout        = x[1]
        Tcell       = x[2]
        pump        = u[0]
        Tsp         = u[1] 
        Trock       = u[2]
        t_free      = u[3]
        pump_o      = u[4]
        Tout_o      = u[5]
        t_free_o    = u[6]

        #8
        #19
        if pump > 0:
            dx[0] += - (Tin - Tsp) / h
            dx[1] += - 0.7 * (Tout - Tin + p[8]) / h
            dx[2] += p[19] * (Tcell - Tin + p[20])

        else:
            dx[0] += p[0] + logistic_function(t_free, p[1:4])
            dx[0] += p[4] * (Tin - Tcell)
            dx[0] += p[5]
            dx[0] += p[6] * (p[7] - Tin)
            
            if t_free < self.t_delay:
                dx[1] += logistic_function(Tout - Tin + p[8], p[16:19])
            else:
                dx[1] += p[9] * (Tout - p[10])
                dx[1] += p[13]
                dx[1] += p[14] * (Tout - Tin - p[15])

            dx[2] += p[21] * (Tcell - Tin + p[22])

        if (pump_o == 1 or t_free_o < self.t_delay) and pump == 0:
            dx[1] += p[11] * (Tout - Tout_o - p[12])

        dx[2] += p[23] * (Tcell - Trock + p[24])

        return dx
    

    def f_p(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.ndarray:
        J = np.zeros((self.Nx, self.Np))

        Tin         = x[0]
        Tout        = x[1]
        Tcell       = x[2]
        pump        = u[0]
        Tsp         = u[1] 
        Trock       = u[2]
        t_free      = u[3]
        pump_o      = u[4]
        Tout_o      = u[5]
        t_free_o    = u[6]

        if pump > 0:
            J[1, 8]     = -0.7 / h
            J[2, 19]    = Tcell - Tin + p[20]
            J[2, 20]    = p[19]

        else:
            J[0, 0]     = 1 
            J[0, 1:4]   = logistic_function_jacobian(t_free, p[1:4])
            J[0, 4]     = Tin - Tcell
            J[0, 5]     = 1
            J[0, 6]     = p[7] - Tin
            J[0, 7]     = p[6]
            
            if t_free < self.t_delay:
                J[1, 8]         = logistic_function_jacobian_state(Tout-Tin+p[8], p[16:19])
                J[1, 16:19]     = logistic_function_jacobian(Tout-Tin+p[8], p[16:19])
            else:
                J[1, 9]     = Tout - p[10]
                J[1, 10]    = -p[9]
                J[1, 13]    = 1
                J[1, 14]    = Tout - Tin - p[15]
                J[1, 15]    = -p[14]
            
            J[2, 21]    = Tcell - Tin + p[22]
            J[2, 22]    = p[21]

        if (pump_o == 1 or t_free_o < self.t_delay) and pump == 0:
            J[1, 11]    = Tout - Tout_o - p[12]
            J[1, 12]    = -p[11]

        J[2, 23]    = Tcell - Trock + p[24]
        J[2, 24]    = p[23]

        return J
    

    def simulate(self, df: pd.DataFrame, p: np.ndarray):
        X = np.zeros((len(df), self.Nx))
        X[0, :] = self.generate_x(df.iloc[0])
        for i in range(len(df)-1):
            u = self.generate_u(df.iloc[i])
            X[i+1, :] = self.f(X[i, :], u, p, 0, self.h)
        return X

