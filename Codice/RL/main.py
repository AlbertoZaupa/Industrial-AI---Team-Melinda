import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from .reward_func import RewardFunction
from .apple_cell import AppleCellEnvironment
from .actor_critic import *
from .buffer import Buffer
from .policy import *

CSV_PATH = "..." # Inserire il path del dataset
PAST_WINDOW = 6*60
NUM_STATES = 2
NUM_ACTIONS = 1
MIN_CELL_TEMP = 0.7
TARGET_CELL_TEMP = 1
MAX_CELL_TEMP = 1.5
MIN_GLYCOL_TEMP = -9
MAX_GLYCOL_TEMP = -4

#CELL_ID = "Ipogeo2_Cella13"
CELL_ID = "Cella_13"
columns = ["TemperaturaCelle", "TemperaturaMandataGlicoleNominale"]
# columns = [CELL_ID + column for column in columns]
DF = pd.read_csv(CSV_PATH)

DF = DF[columns]
DF = DF.replace({',': '.'}, regex=True).astype(float)
DF = DF.fillna(0)

# Vengono utilizzate due copie della rete che sceglie le azioni da compiere
# e due copie della rete che assegna un giudizio alle azioni scelte
actor_model = get_actor(PAST_WINDOW, NUM_STATES, MIN_GLYCOL_TEMP, MAX_GLYCOL_TEMP)
critic_model = get_critic(PAST_WINDOW, NUM_STATES, NUM_ACTIONS)

target_actor = get_actor(PAST_WINDOW, NUM_STATES, MIN_GLYCOL_TEMP, MAX_GLYCOL_TEMP)
target_critic = get_critic(PAST_WINDOW, NUM_STATES, NUM_ACTIONS)

target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate e ottimizzatori
critic_lr = 0.002
actor_lr = 0.001
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Numero totale di episodi di allenamento
total_episodes = 1000
# Per ogni episodio, l'agente sta nella simulazione per un'ora
episode_steps = 60

# Dimensione di una batch di training
batch_size = 64

# Fattore di ammortizzamento per le ricompense future
gamma = 0.99
# Parametro che regola l'update delle copie delle reti
tau = 0.005

# Buffer utilizzato per conservare le esperienze
buffer = Buffer(PAST_WINDOW, NUM_STATES, NUM_ACTIONS, 50000)

reward_function = RewardFunction(target_cell_temp=TARGET_CELL_TEMP,
     min_cell_temp=MIN_CELL_TEMP, max_cell_temp=MAX_CELL_TEMP, 
     min_glycol_temp=MIN_GLYCOL_TEMP, max_glycol_temp=MAX_GLYCOL_TEMP)

env = AppleCellEnvironment("modelli/ED_temp_cella_1min.model", reward_function)


# INIZIA LA FASE DI ALLENAMENTO
avg_reward_list = []

# L'agente inizialmente colleziona un po' di esperienza, compiendo azioni casuali
curr_state = env.reset()
for i in range(10 * batch_size):
  tf_curr_state = tf.expand_dims(tf.convert_to_tensor(curr_state, np.float32), 0)
  action = random_policy(tf_curr_state, MIN_GLYCOL_TEMP, MAX_GLYCOL_TEMP)
  cell_temp, reward, _, __ = env.step(action)
  buffer.record((curr_state, action, reward, cell_temp))
  curr_state = env.update_state(cell_temp, action)

# Fase di allenamento
for ep in range(total_episodes):
    curr_state = env.reset()
    episodic_reward = 0

    for i in range(episode_steps):
      tf_curr_state = tf.expand_dims(tf.convert_to_tensor(curr_state, np.float32), 0)
      action = training_policy(tf_curr_state, actor_model, MIN_GLYCOL_TEMP, MAX_GLYCOL_TEMP)
      # L'ambiente viene modificato applicando l'azione scelta dalla rete, che modifica
      # la temperatura all'interno della cella
      cell_temp, reward, _, __ = env.step(action)

      # Viene salvata la tupla (stato corrente, azione scelta, ricompensa, temperatura della cella)
      buffer.record((curr_state, action, reward, cell_temp))
      episodic_reward += reward
      curr_state = env.update_state(cell_temp, action)

      # Vengono aggiornati i pesi delle reti
      learn(buffer, batch_size)
      # update_target(target_actor.variables, actor_model.variables, tau)
      update_target(target_critic.variables, critic_model.variables, tau)
      
    plt.plot(env.state[:, :1], color="orange")
    plt.plot(env.state[:, 1:], color="blue")
    plt.show()
    
    # Viene salvata la ricompensa media dell'episodio
    avg_reward = episodic_reward / episode_steps
    print("Episode {}, average reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)


plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()