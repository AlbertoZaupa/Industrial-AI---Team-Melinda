import numpy as np


# ReplayBuffer utilizzato per conservare le esperienze passate dell'agente. Queste esperienze vengono poi
# utilizzate per aggiornare i pesi delle reti che compongono l'agente.

class ReplayBuffer:
    def __init__(self, past_window, num_states, time_unit, capacity=1000):
        self.capacity = capacity
        self.counter = 0
        self.time_unit = time_unit
        self.full = False

        self.state_buffer = np.zeros((capacity, past_window, num_states))
        self.action_buffer = np.zeros((capacity, 1, 1))
        self.reward_buffer = np.zeros((capacity, 1, 1))
        self.next_state_buffer = np.zeros((capacity, past_window, num_states))
    
    # In input prende una tupla (s,a,r,s'), che corrisponde ad un'esperienza fatta di:
    #     - uno stato di partenza
    #     - un'azione scelta
    #     - una ricompensa osservata
    #     - uno stato di arrivo

    def record(self, obs_tuple):
        self.state_buffer[self.counter] = obs_tuple[0]
        self.action_buffer[self.counter] = np.reshape(obs_tuple[1], (1, 1)).astype(np.float64)
        self.reward_buffer[self.counter] = np.reshape(obs_tuple[2], (1, 1)).astype(np.float64)
        self.next_state_buffer[self.counter] = obs_tuple[3]

        if self.counter + 1 == self.capacity:
            self.full = True
        
        self.counter = (self.counter + 1) % self.capacity
