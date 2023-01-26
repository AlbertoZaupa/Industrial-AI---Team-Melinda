
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import ceil

from data_importer import translate_to_english
from ode_tableau import RKTableu, RK4_explicit

COLOR_LIST = [(0.0000, 0.4470, 0.7410),
              (0.8500, 0.3250, 0.0980),
              (0.9290, 0.6940, 0.1250),
              (0.4940, 0.6940, 0.5560),
              (0.4660, 0.6740, 0.1880),
              (0.3010, 0.7450, 0.9330),
              (0.6350, 0.0780, 0.1840)]

#  _                                
# | |    ___   __ _  __ _  ___ _ __ 
# | |   / _ \ / _` |/ _` |/ _ \ '__|
# | |__| (_) | (_| | (_| |  __/ |   
# |_____\___/ \__, |\__, |\___|_|   
#             |___/ |___/           
def start_logger():
    f = open("sysid.log", "w")
    f.write("Starting identification problem...\n")
    f.close()

def log_message(msg: str):
    f = open("sysid.log", "a")
    f.write(msg + "\n")
    f.close()


#  __  __           _      _        _
# |  \/  | ___   __| | ___| |   ___| | __ _ ___ ___
# | |\/| |/ _ \ / _` |/ _ \ |  / __| |/ _` / __/ __|
# | |  | | (_) | (_| |  __/ | | (__| | (_| \__ \__ \
# |_|  |_|\___/ \__,_|\___|_|  \___|_|\__,_|___/___/
#
class Model:
    """  
    Base class for the system's model that needs to be identified. Each derived class must set:
    - Nx:           the number of states.
    - Np:           the number of parameters.
    - f:            the continuous-time differential equation that rules the system's dynamics given the 
                    current state x, the input u and the parameters p.
    - f_p:          the Jacobian of f with respect to the parameters p.
    - generate_x:   the function that given the dataframe data, generate the corresponding state 
                    vector.
    - generate_u:   the function that given the dataframe data, generate the corresponding input 
                    vector.
    """

    def __init__(self):
        self.Nx = 0
        self.Nu = 0
        self.Np = 0
        self.h  = 1

    def f(self, x: np.ndarray, u: np.ndarray, p: np.ndarray, t: float, h: float) -> np.ndarray:
        """ Equation of the dynamics """
        pass

    def f_p(self, x: np.ndarray, u: np.ndarray, p: np.ndarray, t: float, h: float) -> np.ndarray:
        """ Jacobian of the dynamics w.r.t. the parameters """
        pass

    def generate_x(self, df: pd.Series) -> np.ndarray:
        """ Given a row of the dataframe, it generates the corresponding state as a vector """
        pass

    def generate_u(self, df: pd.Series) -> np.ndarray:
        """ Given a row of the dataframe, it generates the corresponding input as a vector """
        pass


#  ___    _            _   _  __ _           _   _               ____            _     _
# |_ _|__| | ___ _ __ | |_(_)/ _(_) ___ __ _| |_(_) ___  _ __   |  _ \ _ __ ___ | |__ | | ___ _ __ ___
#  | |/ _` |/ _ \ '_ \| __| | |_| |/ __/ _` | __| |/ _ \| '_ \  | |_) | '__/ _ \| '_ \| |/ _ \ '_ ` _ \
#  | | (_| |  __/ | | | |_| |  _| | (_| (_| | |_| | (_) | | | | |  __/| | | (_) | |_) | |  __/ | | | | |
# |___\__,_|\___|_| |_|\__|_|_| |_|\___\__,_|\__|_|\___/|_| |_| |_|   |_|  \___/|_.__/|_|\___|_| |_| |_|
#
class IdentificationProblem:

    """ 
    Class that defines the identification problem and handles it resolution. Each identification 
    problem is characterized by:
    - data:     the dataframe containing the measured data of the system;
    - model:    the model of the system that needs to be identified;
    - cost:     the cost function that needs to be minimized.
    """

    def __init__(self, data: pd.DataFrame, model: Model):
        self.data = translate_to_english(data)
        self.model = model
        self.ODEsolver = RK4_explicit
        self.show_info = False
        self.fig, self.ax = plt.subplots(2, 1, sharex = True)

    def solve(self, show_plots=False) -> np.ndarray:

        start_logger()

        if show_plots:
            plt.ion()

        p = np.zeros(self.model.Np)

        continue_algorithm = True
        reset_step = 1

        Xref = np.zeros((len(self.data), self.model.Nx))
        for i in range(len(self.data)):
            Xref[i] = self.model.generate_x(self.data.iloc[i])
        xref = Xref.flatten()
        first_update_with_current_step = True
        iter = 1
        th_double_step = 0.001

        while continue_algorithm:
            
            log_message(f"=================================== Iteration {iter: 3d} ===================================")
            log_message(f"Reset steps:       {reset_step: 3d}")
            log_message(f"Current arameters: {p}")

            (Xsim, J) = self.perform_simulation(p, reset_step, return_gradient = True)
            xsim = Xsim.flatten()
            e    = xsim - xref 
            H    = J.T @ J
            dp_s = - np.linalg.solve(H, J.T @ e)
            cost = e.T @ e

            log_message(f"Expected step:     {dp_s}")
            log_message(f"Simulation cost:   {cost}")
            log_message("Performing line-search...")
            dp, c2   = self.line_search(p, dp_s, cost, reset_step, xref)
            log_message("After line-search:")
            log_message(f"Expected step:     {dp}")
            log_message(f"Simulation cost:   {c2}")

            if np.all(dp != dp_s) or cost > c2:
                log_message("Updating parameters")
                p = p + dp
                first_update_with_current_step = False

            if c2 > cost * (1-th_double_step):
                log_message("Low cost improvement, doubling reset step count")
                reset_step = reset_step * 2

            if show_plots:
                self.simulate_parameters(p, reset_step, iter)
                plt.show()

            if reset_step > 300:
                continue_algorithm = False

            iter = iter + 1
        
        if show_plots:
            plt.show(block=True)
        log_message("Finished!")
        return p


    def perform_simulation(self, p: np.ndarray, nreset: int, return_gradient = False):
        """
        Given a set of parameters, it simulates the system based on the stored model and data and it
        compute the cost (that needs to be minimized) and it's gradient w.r.t. the parameters.

        Input:
            - p:    the set of parameters to simulate.

        Returns:
            - cost: the computed cost.
            - J:    the gradient of the cost w.r.t. the parameters.
        """
        # import warnings
        # warnings.filterwarnings("error")

        N       = len(self.data)
        Xsim    = np.zeros((N, self.model.Nx))

        Xsim[0] = self.model.generate_x(self.data.iloc[0])
        x       = self.model.generate_x(self.data.iloc[0])

        if return_gradient:
            J = np.zeros((len(self.data), self.model.Nx, self.model.Np))

        for k in range(N-1):
            
            if k % nreset == 0:
                x = self.model.generate_x(self.data.iloc[k])

            u = self.model.generate_u(self.data.iloc[k])

            if not return_gradient:
                x = self.ODEsolver.integrate(self.model.f, 
                                                x, u, p, 0, self.model.h)
            else:
                x, dJ = self.ODEsolver.integrate(self.model.f,
                                                    x, u, p, 0, self.model.h,
                                                    compute_jacobian=True,
                                                    fun_dp=self.model.f_p)
                J[k] = dJ      

            Xsim[k+1] = x

        if return_gradient:
            return Xsim, J.reshape(self.model.Nx * N, self.model.Np)
        else:
            return Xsim

    def line_search(self, p0, dp, c0, nreset, xm) -> float:
        
        alpha_factor = 1/3
        alpha = 1
        gamma = 0.3
        iter_line_search = 0
        max_iter_line_search = 15
        minc    = c0 
        mindp   = dp

        while iter_line_search < max_iter_line_search:
            
            p    = p0 + alpha * dp
            Xsim = self.perform_simulation(p, nreset, False)
            xsim = Xsim.flatten()
            e    = xsim - xm
            c    = e.T @ e
            log_message(f" > [{int(iter_line_search + 1): 2d}/{max_iter_line_search: 2d}] alpha = {alpha: 1.5f} - cost = {c: 6.3f}")

            if c < minc:
                minc    = c
                mindp   = alpha * dp
            if c < gamma * c0:
                log_message(f" > Succeded! alpha = {alpha: 1.5f}")
                return dp * alpha

            alpha = alpha * alpha_factor        
            iter_line_search = iter_line_search + 1

        log_message(f" > Failed!")
        return mindp, minc


    def simulate_parameters(self, p: np.ndarray, nreset: int = -1, iteration = -1):
        """
        Given a set of parameters, it simulates the system based on the stored model and data.

        Input:
            - p:        the set of parameters to simulate.
            - nreset:   the number of steps after which the state is reset to the initial value.

        Returns:
            - Xsim: the simulated state of the system.
        """
        if nreset == -1:
            nreset = len(self.data)
        X_sim = self.perform_simulation(p, nreset, False)

        X_ref = np.zeros((len(self.data), self.model.Nx))
        U_ref = np.zeros((len(self.data), self.model.Nu))
        for i in range(len(self.data)):
            X_ref[i, :] = self.model.generate_x(self.data.iloc[i])
            U_ref[i, :] = self.model.generate_u(self.data.iloc[i])

        self.ax[0].clear()
        self.ax[1].clear()

        if iteration != -1:
            self.ax[0].set_title(f"Iteration {iteration} - Resetting every {nreset} steps")
        else:
            self.ax[0].set_title("")

        for i in range(self.model.Nx):
            self.ax[0].plot(X_sim[:, i], "-", label=f"x{i+1} simulated", color=COLOR_LIST[i])
            self.ax[0].plot(X_ref[:, i], ".", label=f"x{i+1} data", color=COLOR_LIST[i], ms=0.9)
        self.ax[0].legend()
        self.ax[0].grid()

        for i in range(self.model.Nu):
            self.ax[1].plot(U_ref[:, i], label=f"u{i+1} data")
        self.ax[1].legend()
        self.ax[1].grid()

        plt.draw()
        plt.pause(1)

        return X_sim


