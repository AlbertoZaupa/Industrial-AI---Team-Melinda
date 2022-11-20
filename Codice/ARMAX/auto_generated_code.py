import numpy as np
import pandas as pd


def build_A_matrix(params: list) -> np.ndarray:
	##      States:
	#   0 - TemperaturaMandataGlicoleL0
	#   1 - TemperaturaMandataGlicoleL1
	#   2 - TemperaturaCelleL0
	#   3 - TemperaturaCelleL1
	#   4 - TemperaturaMeleL0
	#   5 - TemperaturaMeleL1
	#   6 - TemperaturaRocciaL0
	A = np.zeros( (7, 7) )

	## Time lag propagation
	A[  1,   0] = 1
	A[  3,   2] = 1
	A[  5,   4] = 1

	## Heat exchange
	A[  0,   0] = 1
	A[  2,   2] = 1
	A[  4,   4] = 1
	A[  6,   6] = 1
	# Dynamics of TemperaturaMandataGlicole due to TemperaturaCelle
	A[  0,   2] += params[  0] # lag 0
	A[  0,   0] -= params[  0]
	A[  0,   3] += params[  1] # lag 1
	A[  0,   0] -= params[  1]
	# Dynamics of TemperaturaCelle due to TemperaturaMandataGlicole
	A[  2,   0] += params[  2] # lag 0
	A[  2,   2] -= params[  2]
	A[  2,   1] += params[  3] # lag 1
	A[  2,   2] -= params[  3]
	# Dynamics of TemperaturaCelle due to TemperaturaMele
	A[  2,   4] += params[  4] # lag 0
	A[  2,   2] -= params[  4]
	A[  2,   5] += params[  5] # lag 1
	A[  2,   2] -= params[  5]
	# Dynamics of TemperaturaMele due to TemperaturaCelle
	A[  4,   2] += params[  6] # lag 0
	A[  4,   4] -= params[  6]
	A[  4,   3] += params[  7] # lag 1
	A[  4,   4] -= params[  7]
	return A


def initialize_states(df: pd.DataFrame) -> np.ndarray:
	x = np.zeros(7)
	x[  0] = df["TemperaturaMandataGlicole"][0]
	x[  2] = df["TemperaturaCelle"][0]
	x[  4] = df["TemperaturaMele"][0]
	x[  6] = df["TemperaturaRoccia"][0]
	return x


def simulate_training(params: list, df: pd.DataFrame) -> np.ndarray:
	X = np.zeros((7, df.shape[0]))
	A = build_A_matrix(params)
	x = initialize_states(df)
	X[:,0] = x
	for i in range(1, df.shape[0]):
		if df["PompaGlicoleMarcia"][i] == 1:
			x[0] = df["TemperaturaMandataGlicoleNominale"][i]
		x = np.dot(A,x)
		for j in range(7):
			if x[j] > 10: x[j] = 10
			if x[j] < -10: x[j] = -10
		X[:,i] = x
	return X


def compute_error(X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
	# Computing error on states with the following weights:
	# TemperaturaMandataGlicole                1.0
	# TemperaturaCelle                         3.0
	# TemperaturaMele                          3.0
	N = df.shape[0]
	e = np.zeros(3*N)
	e[0*N : 1*N] = 1.000000 * (X[0,:] - df["TemperaturaMandataGlicole"])
	e[1*N : 2*N] = 3.000000 * (X[2,:] - df["TemperaturaCelle"])
	e[2*N : 3*N] = 3.000000 * (X[4,:] - df["TemperaturaMele"])
	return e


iteration_counter = 0


def simulate_with_error(pars: list, df: pd.DataFrame) -> np.ndarray:
	global iteration_counter
	print("Iteration", iteration_counter)
	iteration_counter += 1
	X = simulate_training(pars, df)
	return compute_error(X, df)


def MSE_minimization(df: pd.DataFrame, pars0: np.ndarray = np.zeros(8)) -> np.ndarray:
	from scipy.optimize import least_squares
	return least_squares(simulate_with_error, pars0, kwargs={"df":df}, method="trf")


def minimization_result_eval(res, df0: pd.DataFrame) -> pd.DataFrame:
	import scipy

	if type(res) == scipy.optimize.optimize.OptimizeResult:
		pars = res.x
	else:
		pars = res

	X = simulate_training(pars, df0)
	df = pd.DataFrame()
	df["TemperaturaMandataGlicole"] = X[0,:]
	df["TemperaturaCelle"] = X[2,:]
	df["TemperaturaMele"] = X[4,:]
	df["TemperaturaRoccia"] = X[6,:]

	return df


def minimization_compare(df0: pd.DataFrame, df: pd.DataFrame, fig_size=(9,7) ) -> None:
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(3, 1, figsize=fig_size, sharex=True)

	ax[0].plot(df0["Date"], df0["TemperaturaMandataGlicole"], label="true")
	ax[0].plot(df0["Date"], df["TemperaturaMandataGlicole"],  label="predicted")
	ax[0].legend()
	ax[0].grid()
	ax[0].set_title("TemperaturaMandataGlicole")

	ax[1].plot(df0["Date"], df0["TemperaturaCelle"], label="true")
	ax[1].plot(df0["Date"], df["TemperaturaCelle"],  label="predicted")
	ax[1].legend()
	ax[1].grid()
	ax[1].set_title("TemperaturaCelle")

	ax[2].plot(df0["Date"], df0["TemperaturaMele"], label="true")
	ax[2].plot(df0["Date"], df["TemperaturaMele"],  label="predicted")
	ax[2].legend()
	ax[2].grid()
	ax[2].set_title("TemperaturaMele")
