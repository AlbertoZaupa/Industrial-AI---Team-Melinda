import tensorflow as tf


class Config:
    CELL = 13
    TIME_RESOLUTION = 5
    OFF_TIME_REWARD_MULTIPLIER = 3
    PAST_WINDOW_SIZE = 360
    MIN_GLYCOL_TEMP = -7
    MAX_GLYCOL_TEMP = 5

    CSV_PATH = f"../../../datasets/Cella_{CELL}_SH_merged.csv"
    CSV_COLUMNS = ["TemperaturaCelle", "PompaGlicoleMarcia", "TemperaturaMandataGlicole"]
    N_COLUMNS = 3

    TEMP_MODEL_ON_PATH = f"../../../modelli/temp_cell{CELL}_{TIME_RESOLUTION}min__on"
    PUMP_MODEL_ON_PATH = f"../../../modelli/pump_cell{CELL}_{TIME_RESOLUTION}min__on"
    TEMP_MODEL_OFF_PATH = f"../../../modelli/temp_cell{CELL}_{TIME_RESOLUTION}min__off"
    PUMP_MODEL_OFF_PATH = f"../../../modelli/pump_cell{CELL}_{TIME_RESOLUTION}min__off"
    GLYCOL_MODEL_OFF_PATH = f"../../../modelli/glycol_cell{CELL}_{TIME_RESOLUTION}min__off"

    OUTPUT_DIRECTORY = "../../../modelli/agente_rl/"

    DEBUG = True

    REPLAY_BUFFER_SIZE = 500

    # Learning rate e ottimizzatori
    CRITIC_LR = 0.002
    ACTOR_LR = 0.001
    CRITIC_OPTIMIZER = tf.keras.optimizers.Adam(CRITIC_LR)
    ACTOR_OPTIMIZER = tf.keras.optimizers.Adam(ACTOR_LR)

    # Numero totale di episodi di allenamento
    TOTAL_EPISODES = 1000
    # Per ogni episodio, l'agente sta nella simulazione per un'ora
    EPISODE_STEPS = 120
    # Dimensione di una batch di training
    BATCH_SIZE = 64

    # Fattore di ammortizzamento per le ricompense future
    GAMMA = 0.99
    # Parametro che regola l'update delle copie delle reti
    TAU = 0.005

    def __init__(self):
        self.N_COLUMNS = len(self.CSV_COLUMNS)