import numpy as np

# Utilizzata durante la fase di allenamento 
def training_policy(state, actor_model, lower_bound, upper_bound):
    assert upper_bound > lower_bound

    action = actor_model.predict(state, verbose=0)
    sampled_action = tf.squeeze(action)
    # Viene aggiunto del rumore bianco alla temperatura scelta
    mean = 0
    stddev = 0.01
    noise = np.random.normal(mean, stddev, size=1)
    sampled_action = sampled_action.numpy() + noise

    # la temperatura scelta deve sempre stare tra MIN_GLYCOL_TEMP e MAX_GLYCOL_TEMP
    legal_action = np.clip(sampled_action, lower_bound, upper_bound)

    return np.squeeze(legal_action).reshape((NUM_ACTIONS, 1))


# Utilizzata nella fase di esplorazione iniziale
def random_policy(state, lower_bound, upper_bound):
  action = np.random.uniform(lower_bound, upper_bound, size=1)
  return np.squeeze(action).reshape((NUM_ACTIONS, 1))


# Utilizzata al termine dell'allenamento
def final_policy(state, actor_model):
  return tf.squeeze(actor_model.predict(state, verbose=0))