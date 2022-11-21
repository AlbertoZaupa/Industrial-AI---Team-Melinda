import pandas as pd
from reward import *
from environment import *
from agent import *
from config import Config

Config()

if __name__ == "__main__":
    # Viene caricato il dataset
    df = pd.read_csv(Config.CSV_PATH)
    df = df[Config.CSV_COLUMNS]
    df = df.replace({',': '.'}, regex=True).astype(float)
    df = df.fillna(0)

    # L'agente e le reti target vengono inizializzati
    agent = Agent(past_window_size=Config.PAST_WINDOW_SIZE, num_states=Config.N_COLUMNS,
                  lower_bound=-Config.MAX_GLYCOL_DELTA, upper_bound=Config.MAX_GLYCOL_DELTA,
                  buffer_size=Config.REPLAY_BUFFER_SIZE, time_unit=Config.TIME_UNIT, gamma=Config.GAMMA,
                  tau=Config.TAU, lr=Config.LR)
    target_actor = actor_nn(past_window=Config.PAST_WINDOW_SIZE, num_states=Config.N_COLUMNS,
                            lower_bound=-Config.MAX_GLYCOL_DELTA, upper_bound=Config.MAX_GLYCOL_DELTA)
    target_critic = critic_nn(past_window=Config.PAST_WINDOW_SIZE, num_states=Config.N_COLUMNS)
    target_actor.set_weights(agent.actor.get_weights())
    target_critic.set_weights(agent.critic.get_weights())

    # L'ambiente di simulazione
    env = AppleStorageCell(data_source=df, temp_model_on=tf.keras.models.load_model(Config.TEMP_MODEL_ON_PATH),
                           temp_model_off=tf.keras.models.load_model(Config.TEMP_MODEL_OFF_PATH),
                           glycol_model=tf.keras.models.load_model(Config.GLYCOL_MODEL_PATH),
                           reward_func_on=OnRewardFunction(min_glycol_temp=Config.MIN_GLYCOL_TEMP,
                                                           max_glycol_temp=Config.MAX_GLYCOL_TEMP),
                           reward_func_off=OffRewardFunction(), past_window=Config.PAST_WINDOW_SIZE,
                           min_glycol_temp=Config.MIN_GLYCOL_TEMP, max_glycol_temp=Config.MAX_GLYCOL_TEMP,
                           time_unit=Config.TIME_UNIT, temp_setpoint=Config.TEMP_SETPOINT,
                           temp_hysteresis=Config.TEMP_HYSTERESIS, debug=Config.DEBUG)

    # INIZIA LA FASE DI ALLENAMENTO

    # L'agente inizialmente colleziona un po' di esperienza
    curr_state = env.reset_state()
    for i in range(2 * Config.BATCH_SIZE):
        glycol_delta = agent.training_policy(state=curr_state)
        next_state, reward, _ = env.update_state(glycol_delta)
        agent.replay_buffer.record((curr_state, glycol_delta, reward, next_state))
        curr_state = next_state

    # Fase di allenamento
    avg_reward_list = []

    if Config.DEBUG:
        print("training begins")

    for ep in range(Config.TOTAL_EPISODES):
        curr_state = env.reset_state()
        next_state = None
        episodic_reward = 0

        i = 0
        while i < Config.EPISODE_STEPS:
            glycol_delta = agent.training_policy(state=curr_state)
            # L'ambiente viene modificato applicando l'azione scelta dall'agente, che cambia
            # la temperatura all'interno della cella
            next_state, reward, iterations = env.update_state(glycol_delta)
            # Viene salvata la tupla (stato corrente, azione scelta, ricompensa, stato successivo)
            agent.replay_buffer.record((curr_state, glycol_delta, reward, next_state))
            curr_state = next_state
            episodic_reward += reward

            # Gradient descent
            agent.learn(target_actor=target_actor, target_critic=target_critic, batch_size=Config.BATCH_SIZE)
            i += iterations

        if Config.DEBUG:
            env.plot_state(-1)  # (Config.PAST_WINDOW_SIZE)

        # Viene salvata la ricompensa media dell'episodio
        avg_reward = episodic_reward / Config.EPISODE_STEPS
        if Config.DEBUG:
            print(f"Episode {ep}:\naverage reward is ==> {avg_reward}\naverage glycol temperature is ==>"
                  f" {env.state_replay[:, 2:].mean()}\npump duty factor: {env.state_replay[:, 1:2].mean()}")
        avg_reward_list.append(avg_reward)

    plt.plot(avg_reward_list)
    plt.show()

    if input("Save the agent? (Y to save): ") == "Y":
        add_backslash = Config.OUTPUT_DIRECTORY[-1] != "/"
        actor_dir = "/actor" if add_backslash else "actor"
        critic_dir = "/critic" if add_backslash else "critic"
        agent.actor.save(actor_dir)
        agent.critic.save(critic_dir)
