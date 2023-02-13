""" Inlet temperature test model.

This module is used to develop the model of the outlet temperature of the cell lonely.

Info
----
Author: Matteo Dalle Vedove (matteodv99tn@gmail.com)
Date:   07 February 2021
"""

import numpy as np
import pandas as pd

from sysid_problem import Model
from data_importer import import_data 
from sysid_functions import *

STATE_VARIABLES = [
    "outlet_fluid_temperature"
    ]
INPUT_VARIABLES = [
    "pump_status", 
    "inlet_fluid_temperature", 
    "cell_temperature", 
    "rock_temperature", 
    "pump_status_other",
    "outlet_fluid_temperature_other",
    "time_free_run",
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

T_DLY = 15

class Model(Model):


    def __init__(self, h):
        self.Nx = len(STATE_VARIABLES)
        self.Nu = len(INPUT_VARIABLES)
        self.Np = 11
        self.h  = h
        self.name = "Tout"


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

        Tout    = x[0]
        pump    = u[0]
        Tin     = u[1]
        Tcell   = u[2]
        Trock   = u[3]
        pump_o  = u[4]
        Tout_o  = u[5]
        t_free  = u[6]
        t_free_o = u[7]


        if pump > 0:
            dx[0] += - 0.7 * (Tout-Tin + p[0]) / h
            return dx

        if t_free < T_DLY:        
            dx[0] += logistic_function(Tout-Tin + p[0], p[8:11])
            return dx
        else:
            dx[0] += p[1] * (Tout - p[2])
            dx[0] += p[5]
            dx[0] += p[6] * (Tout - Tin - p[7])

        if pump_o == 1 or t_free_o < T_DLY:
            dx[0] += p[3] * (Tout - Tout_o - p[4])

        return dx
    

    def f_p(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.ndarray:
        J = np.zeros((self.Nx, self.Np))

        Tout    = x[0]
        pump    = u[0]
        Tin     = u[1]
        Tcell   = u[2]
        Trock   = u[3]
        pump_o  = u[4]
        Tout_o  = u[5]
        t_free  = u[6]
        t_free_o    = u[7]

        if pump > 0:
            J[0, 0] = -0.7/h
            return J
        
        if t_free < T_DLY:
            J[0, 0]    = logistic_function_jacobian_state(Tout-Tin+p[0], p[8:11])
            J[0, 8:11] = logistic_function_jacobian(Tout-Tin+p[0], p[8:11])
            return J

        else:
            J[0, 1]   = Tout - p[2]
            J[0, 2]   = -p[1]
            J[0, 5]   = 1
            J[0, 6]   = (Tout - Tin - p[7])
            J[0, 7]   = -p[6]
        
        if pump_o == 1 or t_free_o < T_DLY:
            J[0, 3] = Tout - Tout_o - p[4]
            J[0, 4] = -p[3]

        return J
    
    def simulate(self, df: pd.DataFrame, p: np.ndarray):
        X = np.zeros((len(df), self.Nx))
        X[0, :] = self.generate_x(df.iloc[0])
        for i in range(len(df)-1):
            u = self.generate_u(df.iloc[i])
            X[i+1, :] = self.f(X[i, :], u, p, 0, self.h)
        return X

