
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import sysid_config         as cfg

from math           import ceil
from data_importer  import translate_to_english
from ode_tableau    import RKTableu, RK4_explicit
from constants      import SYSID_PATH
from os.path        import join, isfile

LOG_FILE    = join(SYSID_PATH, "sysid.log")
PARAM_FILE  = join(SYSID_PATH, "identified_parameters.log")
EXPORT_FILE = join(SYSID_PATH, "exported_parameters.json")

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
    """Initialize the log file
    
    The file "sysid.log" is created on the directory of the system identification module.
    """
    f = open(LOG_FILE, "w")
    f.write("Starting identification problem...\n")
    f.close()

def log_message(msg: str):
    """Writes a message to the log file
    
    Parameters
    ----------
    msg : str
        The message to be written to the log file.
    """
    f = open(LOG_FILE, "a")
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
    Abstract class for the system's model that needs to be identified.


    Member variables
    ----------------
    Each derived class must set:
    `Nx`: int      
        the number of states of the dynamics.
    `Np`: int       
        the number of parameters to be identified.
    `Nu`: int       
        the number of inputs of the dynamics.
    `h`: float      
        the sampling time of the system for the integration; by default is set to 1.


    Member functions
    ----------------
    Each class must implement:
    - `f`:              the continuous-time differential equation that rules the system's dynamics,
                        so, the function of the form
                        :math:`dx/dt = f(x, u, p, t)`
    - `f_p`:            the analytical Jacobian of the function f with respect to the parameters p.
    - `generate_x`:     the function that given a row of the dataframe, extracts the corresponding
                        ordered state vector.
    - `generate_u`:     the function that given a row of the dataframe, extracts the corresponding
                        ordered input vector for the system's dynamics.

    See the function documentation for furthed details.
    """


    def __init__(self):
        self.Nx = 0
        self.Nu = 0
        self.Np = 0
        self.h  = 1


    def f(self, x: np.ndarray, u: np.ndarray, p: np.ndarray, t: float, h: float) -> np.ndarray:
        """ Equation of the dynamics.
        
        Parameters
        ----------
        x : np.ndarray
            The current state of the system.
        u : np.ndarray
            The current input of the system.
        p : np.ndarray
            The parameters of the dynamics model.
        t : float
            The time of evaluation of the system's dynamics.
        h : float
            Time integration step.

        Returns
        -------
        A numpy array of shape (Nx, 1) containing the continuous time expression of the dynamics, so
        the left-hand side of the expression
            :math:`dx/dt = f(x, u, p, t)`
        """
        pass


    def f_p(self, x: np.ndarray, u: np.ndarray, p: np.ndarray, t: float, h: float) -> np.ndarray:
        """ Jacobian of the dynamics w.r.t. the parameters 
        
        Parameters
        ----------
        x : np.ndarray
            The current state of the system.
        u : np.ndarray
            The current input of the system.
        p : np.ndarray
            The parameters of the dynamics model.
        t : float
            The time of evaluation of the system's dynamics.
        h : float
            Time integration step.

        Returns
        -------
        A numpy array of shape (Nx, Np) containing the Jacobian of the dynamics w.r.t. the the 
        parameters p.
        """
        pass


    def generate_x(self, df: pd.Series) -> np.ndarray:
        """ Generates the measured state vector.
        
        Parameters
        ----------
        df : pd.Series
            A row of the dataframe containing the data to be used for the identification.

        Returns
        -------
        A numpy array of shape (Nx, 1) containing the measured state vector present in the 
        dataframe.
        """
        pass


    def generate_u(self, df: pd.Series) -> np.ndarray:
        """ Generates the measured input vector vector.
        
        Parameters
        ----------
        df : pd.Series
            A row of the dataframe containing the data to be used for the identification.

        Returns
        -------
        A numpy array of shape (Nx, 1) containing the measured state vector present in the 
        dataframe.
        """
        pass


def export_parameters(pars: np.ndarray, mdl: Model) -> None:
    import json 

    if isfile(EXPORT_FILE):
        with open(EXPORT_FILE, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    all_params[mdl.name] = pars.tolist()
    with open(EXPORT_FILE, "w") as f:
        json.dump(all_params, f, indent=4)


def import_parameters(mdl: Model) -> np.ndarray:
    import json
    
    if isfile(EXPORT_FILE):
        with open(EXPORT_FILE, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    if mdl.name in all_params:
        return np.array(all_params[mdl.name])
    else:
        return np.zeros(mdl.Np)


#  ___    _            _   _  __ _           _   _               ____            _     _
# |_ _|__| | ___ _ __ | |_(_)/ _(_) ___ __ _| |_(_) ___  _ __   |  _ \ _ __ ___ | |__ | | ___ _ __ ___
#  | |/ _` |/ _ \ '_ \| __| | |_| |/ __/ _` | __| |/ _ \| '_ \  | |_) | '__/ _ \| '_ \| |/ _ \ '_ ` _ \
#  | | (_| |  __/ | | | |_| |  _| | (_| (_| | |_| | (_) | | | | |  __/| | | (_) | |_) | |  __/ | | | | |
# |___\__,_|\___|_| |_|\__|_|_| |_|\___\__,_|\__|_|\___/|_| |_| |_|   |_|  \___/|_.__/|_|\___|_| |_| |_|
#
class IdentificationProblem:
    """ Class that defines an identification problem.

    Once the parametric model is provided as well as with some data to fit, this class can be used
    to identify the parameters.
    Such problem is regarded as an unconstrained minimization exploiting the Gauss-Newton approach
    improved with a linear search.
    To simplify the simbolic computation, the cost function considered is the sum of the residual 
    squared.

    While solving the problem, a logger file is built "sysid.log" containing informations that can 
    be used for debugging.

    Member variables
    ----------------
    - `data` (pd.DataFrame):    the dataframe containing the measured data of the system;
    - `model` (Model):          the model of the system that needs to be identified;
    - `ODEsolver` (RKTableau):  the integration scheme used for the integration of the dynamics;
    

    Member functions
    ----------------
    - `solve`:                  solves the identification problem and returns the estimated 
                                parameters.
    - `perform_simulation`:     performs a whole simulation of the parametric model using the 
                                provided input data; mainly for internal use.
    - `line_search`:            performs a line search to find the parameter update step; for 
                                internal use only.
    - `simulate_parameters`:    performs a simulation of the parametric model and plots the figure.
    """


    def __init__(self, data: pd.DataFrame, model: Model):
        self.data = translate_to_english(data)
        self.model = model
        self.ODEsolver = RK4_explicit
        self.fig, self.ax = plt.subplots(2, 1, sharex = True)


    def solve(self, p0=None) -> np.ndarray:
        """ Main function to solve the identification problem.
        
        The function exploits the Gauss-Newton approach to solve the problem considering as cost the 
        norm of the residuals squared.

        
        Parameters
        ----------
        p0 : np.ndarray, optional
            The initial guess for the parameters.
            If not provided, a vector of zeros is used.


        Returns
        -------
        A numpy array of shape (Np, 1) containing the identified parameters.
        """

        start_logger()
        if cfg.plot_enabled:
            plt.ion()
            self.solving_problem = True

        if p0 is None:
            p = np.zeros(self.model.Np)
        else:
            p = p0

        mu         = cfg.mu0
        reset_idx  = 0
        reset_step = cfg.reset_steps[reset_idx]

        Xref = np.zeros((len(self.data), self.model.Nx))
        for i in range(len(self.data)):
            Xref[i] = self.model.generate_x(self.data.iloc[i])
        xref = Xref.flatten()
        iter = 1

        while reset_idx < len(cfg.reset_steps):

            reset_step = cfg.reset_steps[reset_idx]

            print(f" > Iteration {iter: 3d}") 
            log_message("==================================="
                        f"Iteration {iter: 3d}" 
                        "===================================")
            log_message(f"Reset steps:       {reset_step: d}")
            log_message(f"Current arameters: {p}")

            (Xsim, J) = self.perform_simulation(p, reset_step, return_gradient = True)
            J    = J / len(self.data)
            xsim = Xsim.flatten()
            e    = xsim - xref 
            H    = J.T @ J 
            dp_s = - np.linalg.solve(H + mu*np.eye(self.model.Np), J.T @ e)
            cost = e.T @ e

            tmp = (H + mu*np.eye(self.model.Np) ) @ J.T
            sv  = np.linalg.svd(tmp, compute_uv=False)

            if np.linalg.norm(dp_s) > 20:
                log_message("Step size exceeds length 20, reducing it")
                dp_s = dp_s / np.linalg.norm(dp_s) * 20

            log_message(f"Expected step:     {dp_s}")
            log_message(f"Simulation cost:   {cost}")
            log_message(f"Mu:                {mu: .2e}")
            log_message(f"Singular values:   max={sv[0]: .2e} - min={sv[-1]: .2e}")
            log_message("Performing line-search...")
            dp, c2   = self.line_search(p, dp_s, cost, reset_step, xref)
            log_message("After line-search:")
            log_message(f"Expected step:     {dp}")
            log_message(f"Simulation cost:   {c2}")

            if np.all(dp != dp_s) or cost > c2:
                log_message("Updating parameters")
                p = p + dp
                if mu > 1e-12:
                    mu = mu * cfg.mu_factor
                first_update_with_current_step = False
            else:
                mu = mu / (cfg.mu_factor**6)

            if c2 > cost * (1-cfg.th_increase_steps):
                log_message("Low cost improvement, doubling reset step count")
                reset_idx = reset_idx + 1

            if cfg.plot_enabled:
                self.simulate_parameters(p, reset_step, iter)
                plt.show()
                
            iter = iter + 1
        
        if cfg.plot_enabled:
            plt.show(block=True)
            self.solving_problem = False

        with open(PARAM_FILE, 'w') as f:
            f.write(str(p))
        log_message("Finished!")
        return p


    def perform_simulation(self, p: np.ndarray, nreset: int, return_gradient = False):
        """
        Given a set of parameters, it simulates the system based on the stored model and data and it
        compute the cost (that needs to be minimized) and it's gradient w.r.t. the parameters.


        Parameters
        ----------
        p: np.ndarray
            The set of parameters to simulate.
        nreset: int
            The number of steps after which the state is reset to the initial measured condition.
        return_gradient: bool
            If True, the gradient of the cost w.r.t. the parameters is also computed. 
            Defaults to false.


        Returns
        -------
        X: np.ndarray
            The simulated state trajectory.
        J: np.ndarray
            The gradient of the cost w.r.t. the parameters. Only returned if return_gradient is the
            corresponding flag is enabled.
        """

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
        """ Performs a line search given a set of a parameters and a search direction.
        
        Parameters
        ----------
        p0: np.ndarray
            The set of parameters to start the line search from.
        dp: np.ndarray
            The search direction.
        c0: float
            The cost of the current parameters p0.
        nreset: int
            The number of steps after which the state is reset to the initial measured condition.
        xm: np.ndarray
            The vector of the measured state trajectory.

        Returns
        -------
        A tuple containing the step to apply, a (Np, 1) array, and the cost of the new parameters.
        """ 

        alpha_factor         = cfg.alpha_factor
        gamma                = cfg.gamma
        alpha                = 1
        iter_line_search     = 0
        max_iter_line_search = 15
        minc                 = c0 * 1e60
        mindp                = dp

        while iter_line_search < cfg.max_linesearch_eval:
            
            p    = p0 + alpha * dp
            Xsim = self.perform_simulation(p, nreset, False)
            xsim = Xsim.flatten()
            e    = xsim - xm
            c    = e.T @ e
            log_message(f" > [{int(iter_line_search + 1): 2d}/{cfg.max_linesearch_eval: 2d}] alpha = {alpha: 1.5f} - cost = {c: 6.3f}")

            if c < minc:
                minc    = c
                mindp   = alpha * dp

            # if c < gamma * c0:
            #     log_message(f" > Succeded! alpha = {alpha: 1.5f}")
            #     return mindp, minc

            if c > minc and c < c0:
                log_message(f" > Starting to increase cost! Stopping at cost {minc}.")
                return mindp, minc

            alpha = alpha * alpha_factor        
            iter_line_search = iter_line_search + 1

        log_message(f" > Failed!")
        return dp, c0


    def simulate_parameters(self, p: np.ndarray, nreset: int = -1, iteration = -1):
        """ Given a set of parameters, it simulates them and provides.

        Parameters
        ----------
        p: np.ndarray
            The set of parameters to simulate.
        nreset: int
            The number of steps after which the state is reset to the initial measured condition.
            Defaults to -1, that is, no state reset is applied.
        iteration: int
            Counter of the iteration inside the minimization, for plotting purposes.
            Defaults to -1, and in this case no title is set to the plot.

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
        self.ax[1].set_ylim([-10.0, 10.0])

        if self.solving_problem:
            plt.draw()
            plt.pause(1)

        return X_sim


