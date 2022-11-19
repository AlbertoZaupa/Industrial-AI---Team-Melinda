import pandas as pd
import matplotlib.pyplot as plt
from reward_func import *
from apple_cell import *
from actor_critic import *
from buffer import ReplayBuffer
from policy import *
from config import Config
Config()

if __name__ == "__main__":
    # Viene caricato il dataset
    df = pd.read_csv(Config.CSV_PATH)
    df = df[Config.CSV_COLUMNS]
    df = df.replace({',': '.'}, regex=True).astype(float)
    df = df.fillna(0)

    # Vengono utilizzate due copie della rete che sceglie le azioni da compiere
    # e due copie della rete che assegna un giudizio alle azioni scelte
    actor_model = get_actor(Config.PAST_WINDOW_SIZE, Config.N_COLUMNS, Config.MIN_GLYCOL_TEMP, Config.MAX_GLYCOL_TEMP)
    critic_model = get_critic(Config.PAST_WINDOW_SIZE, Config.N_COLUMNS, Config.TIME_RESOLUTION)
    target_actor = get_actor(Config.PAST_WINDOW_SIZE, Config.N_COLUMNS, Config.MIN_GLYCOL_TEMP, Config.MAX_GLYCOL_TEMP)
    target_critic = get_critic(Config.PAST_WINDOW_SIZE, Config.N_COLUMNS, Config.TIME_RESOLUTION)
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Replay buffer utilizzato per conservare le esperienze
    buffer = ReplayBuffer(Config.PAST_WINDOW_SIZE, Config.N_COLUMNS, Config.TIME_RESOLUTION, Config.REPLAY_BUFFER_SIZE)

    # L'ambiente di simulazione
    env = AppleStorageCell(data_source=df, temp_model_on=tf.keras.models.load_model(Config.TEMP_MODEL_ON_PATH),
                           pump_model_on=tf.keras.models.load_model(Config.PUMP_MODEL_ON_PATH),
                           temp_model_off=tf.keras.models.load_model(Config.TEMP_MODEL_OFF_PATH),
                           pump_model_off=tf.keras.models.load_model(Config.PUMP_MODEL_OFF_PATH),
                           glycol_model_off=tf.keras.models.load_model(Config.GLYCOL_MODEL_OFF_PATH),
                           reward_func_on=OnRewardFunction(min_glycol_temp=Config.MIN_GLYCOL_TEMP,
                                                           max_glycol_temp=Config.MAX_GLYCOL_TEMP),
                           reward_func_off=OffRewardFunction(Config.OFF_TIME_REWARD_MULTIPLIER, Config.TIME_RESOLUTION),
                           past_window=Config.PAST_WINDOW_SIZE, time_resolution=Config.TIME_RESOLUTION)

    # INIZIA LA FASE DI ALLENAMENTO
    avg_reward_list = []

    # L'agente inizialmente colleziona un po' di esperienza, compiendo azioni casuali
    curr_state = env.reset_state()
    next_state = None
    for i in range(Config.REPLAY_BUFFER_SIZE // 2):
        glycol_temps = random_policy(Config.MIN_GLYCOL_TEMP, Config.MAX_GLYCOL_TEMP, Config.TIME_RESOLUTION)
        next_state, reward = env.update_state(glycol_temps)
        buffer.record((curr_state, glycol_temps, reward, next_state))
        curr_state = next_state

    # Fase di allenamento

    if Config.DEBUG:
        print("training begins")

    for ep in range(Config.TOTAL_EPISODES):
        curr_state = env.reset_state()
        next_state = None
        episodic_reward = 0

        for i in range(Config.EPISODE_STEPS):
            glycol_temps = training_policy(curr_state, actor_model, Config.MIN_GLYCOL_TEMP, Config.MAX_GLYCOL_TEMP)
            # L'ambiente viene modificato applicando l'azione scelta dalla rete, che cambia
            # la temperatura all'interno della cella
            next_state, reward = env.update_state(glycol_temps)
            # Viene salvata la tupla (stato corrente, azione scelta, ricompensa, temperatura della cella)
            buffer.record((curr_state, glycol_temps, reward, next_state))
            episodic_reward += reward.mean()
            curr_state = next_state

            # Vengono aggiornati i pesi delle reti principali (actor_model, critic_model)
            learn(buffer, Config.BATCH_SIZE, target_actor, target_critic, critic_model,
                  actor_model, Config.ACTOR_OPTIMIZER, Config.ACTOR_OPTIMIZER, Config.GAMMA)

            # Vengono aggiornati i pesi delle reti `target` (target_actor, target_critic)
            update_target(target_actor.variables, actor_model.variables, Config.TAU)
            update_target(target_critic.variables, critic_model.variables, Config.TAU)

        if Config.DEBUG:
            plt.title("Cell temperature")
            plt.plot(env.state[0][:, :1], color="orange")
            plt.show()
            plt.title("Pump state")
            plt.plot(env.state[0][:, 1:2], color="red")
            plt.show()
            plt.title("Glycol temperature")
            plt.plot(env.state[0][:, 2:3], color="blue")
            plt.show()

        # Viene salvata la ricompensa media dell'episodio
        avg_reward = episodic_reward / Config.EPISODE_STEPS
        if Config.DEBUG:
            print(f"Episode {ep}, average reward is ==> {avg_reward}")
        avg_reward_list.append(avg_reward)

    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()

    if input("Save the agent?: ") == "Y":
        add_backslash = Config.OUTPUT_DIRECTORY[-1] != "/"
        actor_dir = "/actor" if add_backslash else "actor"
        critic_dir = "/critic" if add_backslash else "critic"
        actor_model.save(actor_dir)
        critic_model.save(critic_dir)
