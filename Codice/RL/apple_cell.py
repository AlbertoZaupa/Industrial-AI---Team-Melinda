import numpy as np
from random import randint


# Questa classe rappresenta l'ambiente simulato delle celle frigorifere
class AppleStorageCell:

    """
    Stato della cella:
        matrice S[past_window][3], colonne:
        - temperatura cella
        - stato pompa
        - temperatura glicole
    """
    TEMP_IDX = 0
    PUMP_IDX = 1
    GLYCOL_IDX = 2

    def __init__(self, data_source, temp_model_on, pump_model_on, temp_model_off, pump_model_off,
                 glycol_model_off, reward_func_on, reward_func_off, past_window, time_resolution):
        self.data_source = data_source
        self.reward_func_on = reward_func_on
        self.reward_func_off = reward_func_off
        self.past_window = past_window
        self.time_resolution = time_resolution

        # i modelli per la simulazione
        self.temp_model_on = temp_model_on
        self.pump_model_on = pump_model_on
        self.temp_model_off = temp_model_off
        self.pump_model_off = pump_model_off
        self.glycol_model_off = glycol_model_off

        self.state = None

    def reset_state(self):
        start = randint(0, self.data_source.shape[0] - self.past_window)
        self.state = self.data_source.iloc[start:start + self.past_window].values
        self.state = np.reshape(self.state, (1, self.past_window, self.data_source.shape[1]))
        return self.state

    def update_state(self, glycol_temps):
        if self.is_refrigeration_on():
            self.update_state_on(glycol_temps)
            reward = self.reward_func_on(glycol_temps[0])
        else:
            off_time = self.update_state_off()
            reward = self.reward_func_off(off_time)
        return self.state, reward

    def update_state_on(self, future_glycol_temps):
        future_cell_temps = self.temp_model_on.predict((self.state, future_glycol_temps), verbose=0)[0]
        future_pump_states = np.around(self.pump_model_on.predict((self.state, future_glycol_temps), verbose=0)[0])
        future_glycol_temps = np.reshape(future_glycol_temps, (self.time_resolution, 1))
        state_update = np.concatenate((future_cell_temps, future_pump_states, future_glycol_temps), axis=1)

        self.state = np.concatenate((self.state[0][self.time_resolution:], state_update), axis=0)
        self.state = np.reshape(self.state, (1, self.state.shape[0], self.state.shape[1]))

    def update_state_off(self):
        off_time = 0
        while not self.is_refrigeration_on():
            future_cell_temps = self.temp_model_off.predict(self.state, verbose=0)[0]
            future_pump_states = np.around(self.pump_model_off.predict(self.state, verbose=0)[0])
            future_glycol_temps = self.glycol_model_off.predict(self.state, verbose=0)[0]
            state_update = np.concatenate((future_cell_temps, future_pump_states, future_glycol_temps), axis=1)

            self.state = np.concatenate((self.state[0][self.time_resolution:], state_update), axis=0)
            self.state = np.reshape(self.state, (1, self.state.shape[0], self.state.shape[1]))
            off_time += self.time_resolution
        return off_time

    def is_refrigeration_on(self):
        return self.state[-self.time_resolution:, self.PUMP_IDX:self.PUMP_IDX + 1].mean() > 0.5
