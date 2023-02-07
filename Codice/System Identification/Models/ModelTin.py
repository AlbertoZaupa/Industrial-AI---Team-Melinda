""" Inlet temperature test model.

This module is used to develop the model of the inlet temperature of the cell lonely.

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
    "inlet_fluid_temperature"
    ]
INPUT_VARIABLES = [
    "pump_status", 
    "inlet_setpoint_temperature", 
    "rock_temperature", 
    "cell_temperature", 
    "outlet_fluid_temperature", 
    "time_free_run"
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

    for i in range(1, len(df)):
        if df.pump_status[i] == 1:
            df.time_free_run[i] = 0
        else:
            df.time_free_run[i] = df.time_free_run[i-1] + 1

    return df


class Model(Model):


    def __init__(self, h):
        self.Nx = len(STATE_VARIABLES)
        self.Nu = len(INPUT_VARIABLES)
        self.Np = 8
        self.h  = h


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

        Tin     = x[0]
        pump    = u[0]
        Tsp     = u[1]
        Trock   = u[2]
        Tcell   = u[3]
        Tout    = u[4]
        t       = u[5]

        if pump > 0:    # pump  on
            dx[0] = - (Tin-Tsp)/h
        else:
            dx[0] += p[0] + logistic_function(t, p[1:4])
            dx[0] += p[4] * (Tin - Tcell)
            dx[0] += p[5]
            dx[0] += p[6] * (p[7] - Tin)

        return dx
    

    def f_p(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.ndarray:
        
        J = np.zeros((self.Nx, self.Np))

        Tin     = x[0]
        pump    = u[0]
        Tcell   = u[3]
        t       = u[5]

        if pump == 0:
            J[0, 0]     = 1
            J[0, 1:4]   = logistic_function_jacobian(t, p[1:4])
            J[0, 4]     = Tin - Tcell
            J[0, 5]     = 1
            J[0, 6]     = p[7] - Tin
            J[0, 7]     = p[6]
        return J
    
    def simulate(self, df: pd.DataFrame, p: np.ndarray):
        X = np.zeros((len(df), self.Nx))
        X[0, :] = self.generate_x(df.iloc[0])
        for i in range(len(df)-1):
            u = self.generate_u(df.iloc[i])
            X[i+1, :] = self.f(X[i, :], u, p, 0, self.h)
        return X

