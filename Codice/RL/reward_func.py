import numpy as np


class OnRewardFunction:

    def __init__(self, min_glycol_temp, max_glycol_temp):

        assert max_glycol_temp > min_glycol_temp

        # temperature massima e minima del glicole
        self.min_glycol_temp = min_glycol_temp
        self.max_glycol_temp = max_glycol_temp
        # fattore di scala per calcolare la penalità associata ad un determinato valore di temperatura del glicole
        self.glycol_range = max_glycol_temp - min_glycol_temp

    def __call__(self, glycol_temps):
        # """ Penalità associata ad un determinato valore di temperatura del glicole """
        return (self.min_glycol_temp - glycol_temps) / self.min_glycol_temp


class OffRewardFunction:

    def __init__(self, reward_multiplier, output_size):
        self.reward_multiplier = reward_multiplier
        self.output_size = output_size

    def __call__(self, off_time):
        return np.ones((self.output_size, 1)) * off_time * self.reward_multiplier
