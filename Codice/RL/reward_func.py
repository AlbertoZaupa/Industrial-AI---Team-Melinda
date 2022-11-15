import numpy as np
from math import isnan

class RewardFunction:

    def __init__(self, target_cell_temp, min_cell_temp,
       max_cell_temp, min_glycol_temp,
       max_glycol_temp, lookback_window=100):

        assert max_glycol_temp > min_glycol_temp
        assert max_cell_temp > min_cell_temp
        assert target_cell_temp > min_cell_temp
        assert target_cell_temp < max_cell_temp
        assert lookback_window > 0

        self.min_cell_temp = min_cell_temp # minima temperatura tollerata all'interno della cella
        self.max_cell_temp = max_cell_temp # massima temperatura tollerata all'interno della cella
        self.target_cell_temp = target_cell_temp # temperatura desiderata all'interno della cella
        # fattore di scala per calcolare la penalità associata ad una determinata temperatura
        self.cell_temp_range = max(max_cell_temp - target_cell_temp, target_cell_temp - min_cell_temp)

        # solo la temperatura minima del glicole viene salvata, non ci serve salvare la temperatura massima
        self.max_glycol_temp = max_glycol_temp
        # fattore di scala per calcolare la penalità associata ad un determinato valore di temperatura del glicole
        self.glycol_range = max_glycol_temp - min_glycol_temp

        self.lookback_window = lookback_window
        self.lookback_array = np.array([])
        self.lookback_count = 0
        self.lookback_sum = 0

    def glycol_temp_penalty(self, glycol_temp):
        """ Penalità associata ad un determinato valore di temperatura del glicole """
        if self.lookback_count < self.lookback_window:
            self.lookback_sum = self.lookback_sum + glycol_temp
            self.lookback_array = np.concatenate((self.lookback_array, glycol_temp[0]))
        else:
            self.lookback_sum = self.lookback_sum - self.lookback_array[0] + glycol_temp
            self.lookback_array = np.concatenate((self.lookback_array[1:], glycol_temp[0]))

        self.lookback_count = min(self.lookback_count + 1, self.lookback_window)
        penalty = ((self.lookback_sum / self.lookback_count) - self.max_glycol_temp)/self.glycol_range
        if isnan(penalty):
            print(f"Chosen glycol temp: {glycol_temp}")
            print(f"Lookback sum: {self.lookback_sum}")
            print(f"Lookback count: {self.lookback_count}")
            print(f"Max glycol temp: {self.max_glycol_temp}")
            print(f"Glycol range: {self.glycol_range}")
            print(f"Lookback window: {self.lookback_window}")
            print(f"Lookback array:\n {self.lookback_array}")
        return penalty

    def cell_temp_penalty(self, cell_temp):
        """ Penalità associata ad un determinato valore di temperatura all'interno della cella """

        penalty = abs(cell_temp - self.target_cell_temp)/self.cell_temp_range
        if isnan(penalty):
            print(f"Actual cell temp: {cell_temp}")
            print(f"Desired cell temp: {self.target_cell_temp}")
            print(f"Cell temp range: {self.cell_temp_range}")
        return penalty
  
    def __call__(self, cell_temp, glycol_temp):
        """
          Ritorna la penalità associata ad una coppia (temp_cella, temp_glicole).
          I valori di penalità sono nell'intervallo [-2, 0].
          Le penalità associate a temperatura della cella e del glicole variano entrambe nel range [-1, 0].
        """

        if cell_temp < self.min_cell_temp or cell_temp > self.max_cell_temp:
            return -2

        glycol_penalty = self.glycol_temp_penalty(glycol_temp)
        cell_penalty = - self.cell_temp_penalty(cell_temp)
        if isnan(glycol_penalty) or isnan(cell_penalty):
            assert False
        return glycol_penalty + cell_penalty
