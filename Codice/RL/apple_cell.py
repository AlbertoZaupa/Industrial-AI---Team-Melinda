import numpy as np
import tensorflow as tf
from random import randint
import gym


# Funzione utilizzata per determinare lo stato di partenza della simulazione, che
# viene scelto a caso dal dataset
def draw_random_initial_state(df, past_window):
    start = randint(0, df.shape[0] - past_window)
    return df.iloc[start:start + past_window].values


# Questa classe rappresenta l'ambiente simulato delle celle frigorifere
class AppleCellEnvironment(gym.Env):

    def __init__(self, model_path, reward_function, df, past_window):
        super(AppleCellEnvironment, self).__init__()
        self.state = None
        self.model = tf.keras.models.load_model(model_path)  # la rete neurale utilizzata per la simulazione
        self.reward_function = reward_function  # la funzione di ricompensa/penalità
        self.df = df
        self.past_window = past_window

    def step(self, glycol_temp):
        future_cell_temp = self.model.predict(
            (tf.reshape(self.state, (1, self.state.shape[0], self.state.shape[1])),
             tf.reshape(glycol_temp, (1, glycol_temp.shape[0], glycol_temp.shape[1]))), verbose=0)  # la temperatura futura della cella viene calcolata
        reward = self.reward_function(future_cell_temp, glycol_temp)  # viene calcolata la ricompensa

        return future_cell_temp, reward, False, {}

    def reset(self):
        self.state = draw_random_initial_state(self.df, self.past_window)
        return self.state

    def update_state(self, future_cell_temp, glycol_temp):
        # I nuovi valori delle temperature vengono salvati, mentre il campione più vecchio viene dimenticato
        self.state = tf.convert_to_tensor(np.concatenate((self.state[1:], np.array([future_cell_temp, glycol_temp])
                                                          .reshape((1, 2))), axis=0), np.float32)
        return self.state

    # Non è necessari implementare questa funzione
    def render(self):
        pass