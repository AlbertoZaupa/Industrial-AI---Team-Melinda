import numpy as np


# Utilizzata durante la fase di allenamento 
def training_policy(state, actor_model, lower_bound, upper_bound):
    assert upper_bound > lower_bound
    action = actor_model.predict(state, verbose=0)
    # Viene aggiunto del rumore bianco alla temperatura scelta
    mean = 0
    stddev = 1
    noise = np.random.normal(mean, stddev, size=1)
    action = action + noise

    # la temperatura scelta deve sempre stare tra MIN_GLYCOL_TEMP e MAX_GLYCOL_TEMP
    clip = np.clip(action, lower_bound, upper_bound)
    return clip


# Utilizzata nella fase di esplorazione iniziale
def random_policy(lower_bound, upper_bound, time_resolution):
    return np.ones((1, time_resolution, 1)) * np.random.uniform(lower_bound, upper_bound, size=1)


# Utilizzata al termine dell'allenamento
def final_policy(state, actor_model):
    return actor_model.predict(state, verbose=0)
