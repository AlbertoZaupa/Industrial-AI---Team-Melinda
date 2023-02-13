
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sysid_problem      import IdentificationProblem, import_parameters, export_parameters
from data_importer      import import_data, show_cell_data
from Models.Model1      import Model, import_model_data
from sysid_functions    import *

PLOT_DATA = False
PERFORM_MINIMIZATION = True

df      = import_model_data(13)
red_df  = df[6300: -6500]
mdl     = Model(1)
problem = IdentificationProblem(red_df, mdl)


if PLOT_DATA:
    df_plot = import_data(13, False, True)
    show_cell_data(df_plot, df_plot.date[0], df_plot.date[len(df_plot)-1])
    plt.show()


if PERFORM_MINIMIZATION:
    
    print("Solving the problem...")
    if mdl.name != "Model1":
        pars = import_parameters(mdl)
    else:
        pars =np.concatenate((import_parameters("Tin"),
                              import_parameters("Tout"),
                              import_parameters("Tcell")))    
    print(pars)
    p = problem.solve(pars)
    export_parameters(p, mdl)
    print("Problem solved!")
    print("Final parameters: ", p)

else:
    p = import_parameters(mdl)

problem.simulate_parameters(p)
plt.show()
