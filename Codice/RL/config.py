class Config:
    CELL = 13
    TIME_UNIT = 5
    PAST_WINDOW_SIZE = 360
    MIN_GLYCOL_TEMP = -7
    MAX_GLYCOL_TEMP = 4
    MAX_GLYCOL_DELTA = 0.3
    TEMP_HYSTERESIS = 1.3
    TEMP_SETPOINT = 0.8

    CSV_PATH = f"../../../datasets/Cella_{CELL}_SH_merged.csv"
    CSV_COLUMNS = ["TemperaturaCelle", "PompaGlicoleMarcia", "TemperaturaMandataGlicole"]
    N_COLUMNS = 3

    TEMP_MODEL_ON_PATH = f"../../../modelli/temp_cell{CELL}_{TIME_UNIT}min__on"
    TEMP_MODEL_OFF_PATH = f"../../../modelli/temp_cell{CELL}_{TIME_UNIT}min__off"
    GLYCOL_MODEL_PATH = f"../../../modelli/glycol_cell{CELL}_{TIME_UNIT}min__off"

    OUTPUT_DIRECTORY = "../../../modelli/agente_rl/"

    DEBUG = True

    REPLAY_BUFFER_SIZE = 5000

    # Learning rate
    LR = 0.001

    # Numero totale di episodi di allenamento
    TOTAL_EPISODES = 100
    # Per ogni episodio, l'agente sta nella simulazione per un'ora
    EPISODE_STEPS = 240
    # Dimensione di una batch di training
    BATCH_SIZE = 64

    # Fattore di ammortizzamento per le ricompense future
    GAMMA = 0.99
    # Parametro che regola l'update delle copie delle reti
    TAU = 0.005

    def __init__(self):
        self.N_COLUMNS = len(self.CSV_COLUMNS)
