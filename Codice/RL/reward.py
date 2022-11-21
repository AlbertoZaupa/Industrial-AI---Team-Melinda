class OnRewardFunction:

    def __init__(self, min_glycol_temp, max_glycol_temp):
        assert max_glycol_temp > min_glycol_temp

        # temperature massima e minima del glicole
        self.min_glycol_temp = min_glycol_temp
        self.max_glycol_temp = max_glycol_temp
        # fattore di scala per calcolare la penalità associata ad un determinato valore di temperatura del glicole
        self.glycol_range = max_glycol_temp - min_glycol_temp

    def __call__(self, glycol_temp):
        # Penalità associata ad un determinato valore di temperatura del glicole
        return (glycol_temp - self.max_glycol_temp) / self.glycol_range


class OffRewardFunction:

    def __call__(self, off_time):
        return off_time
