
import numpy    as np
import pandas   as pd
from data_importer import translate_to_english
from ode_tableau import RKTableu, RK4_explicit

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
        self.Np = 0


    def f(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.array:
        """ Equation of the dynamics """
        return np.zeros(self.Nx)
    

    def f_p(self, x: np.array, u: np.array, p: np.array, t: float, h: float) -> np.ndarray:
        """ Jacobian of the dynamics w.r.t. the parameters """
        return np.zeros((self.Nx, self.Np))


    def generate_x(self, df: pd.Series) -> np.array:
        """ Given a row of the dataframe, it generates the corresponding state as a vector """
        return np.zeros(self.Nx)


    def generate_u(self, df: pd.Series) -> np.array:
        """ Given a row of the dataframe, it generates the corresponding input as a vector """
        return np.zeros(self.Nx)


#   ____          _          _               
#  / ___|___  ___| |_    ___| | __ _ ___ ___ 
# | |   / _ \/ __| __|  / __| |/ _` / __/ __|
# | |__| (_) \__ \ |_  | (__| | (_| \__ \__ \
#  \____\___/|___/\__|  \___|_|\__,_|___/___/
#
class Cost: 
    """
    Parent class for the definition of the cost function that needs to be minimized.
    """                                            

    def __init__(self):
        pass


    def cost(self, x_meas: np.array, x: np.array) -> float:
        """ 
        Cost function 
        
        Parameters:
        - x_meas:   the measured states coming from the real system's data.
        - x:        the estimated states coming from the simulation and the given set of parameters.

        Returns:    the cost.
        """
        return 0.0


    def cost_dx(self, x_meas: np.array, x: np.array) -> np.array:
        """ 
        Gradient of the cost function w.r.t. the estimated states

        
        Parameters:
        - x_meas:   the measured states coming from the real system's data.
        - x:        the estimated states coming from the simulation and the given set of parameters.

        Returns:    the gradient of the cost function w.r.t. the estimated states x.
        """
        return np.zeros(x_meas.shape)


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

    def __init__(self, data: pd.DataFrame, model: Model, cost: Cost):
        self.data  = translate_to_english(data)
        self.model = model
        self.cost  = cost 
        self.ODEsolver = RK4_explicit

    
    def solve(self, p0: np.array = None) -> np.array:

        if p0 is None:
            p0 = np.zeros(self.model.Np)

        return p0

    
    def compute_cost_and_gradient(self, p: np.array):
        """
        Given a set of parameters, it simulates the system based on the stored model and data and it
        compute the cost (that needs to be minimized) and it's gradient w.r.t. the parameters.

        Input:
            - p:    the set of parameters to simulate.
        
        Returns:
            - cost: the computed cost.
            - J:    the gradient of the cost w.r.t. the parameters.
        """


        T           = self.data.iloc[-1].minutes                # simulation time
        h           = 1;                                        # time step 
        t_vect      = np.arange(h, T, h)                        # time vector of simulation
        cost        = 0.0                                       # cost
        J           = np.zeros(self.model.Np)                   # jacobian of the cost w.r.t. the 
                                                                # parameters
        data_idx    = 1                                         # index of the data frame

        x           = self.model.generate_x(self.data.iloc[0])  # initial state
        u           = self.model.generate_u(self.data.iloc[0])  # initial input

        for t in t_vect:

            if self.data.minutes.iloc[data_idx] == t:
                u = self.model.generate_u(self.data.iloc[data_idx])

            (xnext, dJ) = self.ODEsolver.integrate( self.model.f, 
                                                    x, u, p, t, h,
                                                    compute_jacobian = True,
                                                    fun_dp = self.model.f_p
                                                    )
            x = xnext

            if self.data.minutes.iloc[data_idx] == t:                
                x_meas       = self.model.generate_x(self.data.iloc[data_idx])
                cost        += self.cost.cost(x_meas, x)
                J           += self.cost.cost_dx(x_meas, x) @ dJ
                data_idx    += 1

        return cost, J


