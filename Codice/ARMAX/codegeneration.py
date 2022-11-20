"""
Questo file viene utilizzato per creare il fiel "auto_generated_code.py" che 
serve per effettuare l'allenamento del sistema ibrido lineare. Per questo motivo
è necessario definire nella cartella "CodeGenInfo" i seguenti file:

  - "states.txt": contiene la lista degli stati da considerare e il numero di 
    lag temporali considerati (considerando anche quello corrente). Esempio:
        TemperaturaMandataGlicole 1

  - "heat_exchange.txt": contiene la lista degli accoppiamenti di scambio di 
    temperatura. Considerando per esempio una riga
        TemperaturaMandataGlicole TemperaturaCella
    significa che la temperatura in mandata del glicole è influenzata dalla 
    temperatura della cella (e tutti i suoi lag precedenti)

  - "error_weights.txt": contiene la lista degli stati del quale valutare l'
    errore e il peso associato.

Attenzione: questo codice è stato un "flusso di coscienza", quindi non è scritto
bene e la formattazione non è il massimo, ma fa il suo lavoro.
"""

import os.path as osp

num_to_state = {}
state_to_num = {}
state_recursive = {}
max_lag = 0
i = 0

def compute_state_name(name, lag):
    return name + f"L{int(lag)}"

def initialize_file(filename):
    with open(filename, 'w') as file:
        file.write('import numpy as np\n')
        file.write('import pandas as pd\n')
        file.write('\n')

with open(osp.join('Codice', 'ARMAX', 'CodeGenInfo','states.txt'), 'r') as file:
    lines = file.readlines()
    for line in lines:
        state_name = line.split()[0]
        state_rec  = int(line.split()[-1])
        if state_rec > max_lag:
            max_lag = state_rec
        num_to_state.update({i: state_name})
        state_to_num.update({state_name: i})
        state_recursive.update({state_name: state_rec})
        i = i+1

state_relations = list()
with open(osp.join('Codice', 'ARMAX', 'CodeGenInfo','heat_exchange.txt'), 'r') as file:
    lines = file.readlines()
    for line in lines:
        s1 = line.split()[0]
        s2 = line.split()[-1]
        state_relations.append( (s1, s2) )

state_errors = {}
with open(osp.join('Codice', 'ARMAX', 'CodeGenInfo','error_weights.txt'), 'r') as file:
    lines = file.readlines()
    for line in lines:
        s1 = line.split()[0]
        s2 = line.split()[-1]
        state_errors.update({s1: float(s2)})

i = 0
states = {}
for state in state_recursive:
    lagcount = state_recursive.get(state)
    for j in range(0, lagcount):
        newstate = compute_state_name(state, j)
        states.update({newstate: i})
        i = i+1

myfile = osp.join('Codice','ARMAX', 'auto_generated_code.py')
par_counter = 0
initialize_file(myfile)
with open(myfile, 'a') as file:
    file.write('\ndef build_A_matrix(params: list) -> np.ndarray:\n')
    file.write('\t##      States:\n')
    for s in states:
        file.write(f'\t# {states.get(s):3d} - {s}\n')
    file.write(f'\tA = np.zeros( ({len(states)}, {len(states)}) )\n')
    file.write('\n\t## Time lag propagation\n')
    for s in state_recursive:
        for j in range(1, state_recursive.get(s)):
            curr = compute_state_name(s, j)
            prev = compute_state_name(s, j-1)
            file.write(f'\tA[{states.get(curr):3d}, {states.get(prev):3d}] = 1\n')
            
    file.write('\n\t## Heat exchange\n')
    for state in state_recursive:
        file.write(f'\tA[{states.get(state+"L0"):3d}, {states.get(state+"L0"):3d}] = 1\n')
    for state_pair in state_relations:
        s1 = state_pair[0]
        s2 = state_pair[1]

        file.write(f'\t# Dynamics of {s1} due to {s2}\n')

        target = compute_state_name(s1, 0)
        for j in range(state_recursive.get(s2)):
            start = compute_state_name(s2, j)
            file.write(f'\tA[{states.get(target):3d}, {states.get(start):3d}] += params[{par_counter:3d}] # lag {j}\n')
            file.write(f'\tA[{states.get(target):3d}, {states.get(target):3d}] -= params[{par_counter:3d}]\n')
            par_counter = par_counter + 1

    file.write('\treturn A\n')

    
    file.write('\n\ndef initialize_states(df: pd.DataFrame) -> np.ndarray:\n')
    file.write(f'\tx = np.zeros({len(states)})\n')
    for state in state_recursive:
        state_name = compute_state_name(state, 0)
        file.write(f'\tx[{states.get(state_name):3d}] = df["{state}"][0]\n')
    file.write('\treturn x\n')

    
    file.write('\n\ndef simulate_training(params: list, df: pd.DataFrame) -> np.ndarray:\n')
    file.write(f'\tX = np.zeros(({len(states)}, df.shape[0]))\n')
    file.write('\tA = build_A_matrix(params)\n')
    file.write(f'\tx = initialize_states(df)\n')
    file.write('\tX[:,0] = x\n')
    file.write('\tfor i in range(1, df.shape[0]):\n')
    file.write('\t\tif df["PompaGlicoleMarcia"][i] == 1:\n')
    file.write(f'\t\t\tx[{states.get("TemperaturaMandataGlicoleL0")}] = df["TemperaturaMandataGlicoleNominale"][i]\n')
    file.write('\t\tx = np.dot(A,x)\n')
    file.write(f'\t\tfor j in range({len(states)}):\n')
    file.write(f'\t\t\tif x[j] > 10: x[j] = 10\n')
    file.write(f'\t\t\tif x[j] < -10: x[j] = -10\n')
    file.write('\t\tX[:,i] = x\n')
    file.write('\treturn X\n')

    
    file.write('\n\ndef compute_error(X: np.ndarray, df: pd.DataFrame) -> np.ndarray:\n')
    file.write('\t# Computing error on states with the following weights:\n')
    for state in state_errors:
        file.write(f'\t# {state.ljust(40, " ")} {state_errors.get(state)}\n')
    file.write('\tN = df.shape[0]\n')
    file.write(f'\te = np.zeros({len(state_errors)}*N)\n')
    for i, state in enumerate(state_errors):
        file.write(f'\te[{i}*N : {i+1}*N] = {state_errors.get(state):5f} * (X[{states.get(state+"L0")},:] - df["{state}"])\n')
    file.write('\treturn e\n')
    
    file.write('\n\niteration_counter = 0\n')
    file.write('\n\ndef simulate_with_error(pars: list, df: pd.DataFrame) -> np.ndarray:\n')
    file.write('\tglobal iteration_counter\n')
    file.write('\tprint("Iteration", iteration_counter)\n')
    file.write('\titeration_counter += 1\n')
    file.write('\tX = simulate_training(pars, df)\n')
    file.write('\treturn compute_error(X, df)\n')


    file.write(f'\n\ndef MSE_minimization(df: pd.DataFrame, pars0: np.ndarray = np.zeros({par_counter})) -> np.ndarray:\n')
    file.write('\tfrom scipy.optimize import least_squares\n')
    file.write('\treturn least_squares(simulate_with_error, pars0, kwargs={"df":df}, method="trf")\n')

    file.write(f'\n\ndef minimization_result_eval(res, df0: pd.DataFrame) -> pd.DataFrame:\n')
    file.write('\timport scipy\n\n')
    file.write('\tif type(res) == scipy.optimize.optimize.OptimizeResult:\n')
    file.write('\t\tpars = res.x\n')
    file.write('\telse:\n')
    file.write('\t\tpars = res\n\n')
    file.write('\tX = simulate_training(pars, df0)\n')
    file.write('\tdf = pd.DataFrame()\n')
    for state in state_recursive:
        file.write(f'\tdf["{state}"] = X[{states.get(state+"L0")},:]\n')
    file.write('\n\treturn df\n')

    file.write(f'\n\ndef minimization_compare(df0: pd.DataFrame, df: pd.DataFrame, fig_size=(9,7) ) -> None:\n')
    file.write('\timport matplotlib.pyplot as plt\n')
    file.write(f'\tfig, ax = plt.subplots({len(state_errors)}, 1, figsize=fig_size, sharex=True)\n')
    for ind, state in enumerate(state_errors):
        file.write(f'\n\tax[{ind}].plot(df0["Date"], df0["{state}"], label="true")\n')
        file.write(f'\tax[{ind}].plot(df0["Date"], df["{state}"],  label="predicted")\n')
        file.write(f'\tax[{ind}].legend()\n')
        file.write(f'\tax[{ind}].grid()\n')
        file.write(f'\tax[{ind}].set_title("{state}")\n')


# least_squares(MSE, np.ones(6), kwargs={'df':df}, method='lm')