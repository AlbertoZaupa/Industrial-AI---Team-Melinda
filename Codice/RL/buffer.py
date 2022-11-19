import numpy as np


# ReplayBuffer circolare utilizzato per conservare le esperienze passate dell'agente
class ReplayBuffer:
    def __init__(self, past_window, num_states, time_resolution, capacity=10000):
        self.capacity = capacity
        self.counter = 0
        self.time_resolution = time_resolution
        self.full = False

        self.state_buffer = np.zeros((capacity, past_window, num_states))
        self.action_buffer = np.zeros((capacity, time_resolution, 1))
        self.reward_buffer = np.zeros((capacity, time_resolution, 1))
        self.next_state_buffer = np.zeros((capacity, past_window, num_states))
    
    # In input prende una tupla (s,a,r,s'), che corrisponde ad un'esperienza
    def record(self, obs_tuple):
        self.state_buffer[self.counter] = obs_tuple[0]
        self.action_buffer[self.counter] = obs_tuple[1]
        self.reward_buffer[self.counter] = obs_tuple[2]
        self.next_state_buffer[self.counter] = obs_tuple[3]

        if self.counter + 1 == self.capacity:
            self.full = True
        
        self.counter = (self.counter + 1) % self.capacity
