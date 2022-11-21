class Config:
    # Cella scelta
    CELL = 13

    # Durata (in minuti) di uno step della simulazione
    TIME_UNIT = 5

    # Numero di campioni passati che costituisce lo stato del sistema. Il valore
    # deve essere uguale al numero di campioni passati che prendono in input le reti
    # utilizzate per la simulazione
    PAST_WINDOW_SIZE = 360

    # Temperature minima e massima accettabili per il glicole
    MIN_GLYCOL_TEMP = -7
    MAX_GLYCOL_TEMP = 4

    # Massimo incremento possibile della temperatra del glicole tra uno step della simulazione ed
    # il successivo
    MAX_GLYCOL_DELTA = 0.3

    # Temperatura di isteresi e di setpoint, necessarie per determinare lo stato della pompa.
    # Non avendo avuto a disposizione questi valori nel dataset, li abbiamo impostati come valori
    # costanti.
    TEMP_HYSTERESIS = 1.3
    TEMP_SETPOINT = 0.8

    # Path al dataset
    CSV_PATH = f"../../../datasets/Cella_{CELL}_SH_merged.csv"

    # Le colonne che costituiscono lo stato della cella nella simulazione. L'ordine in cui le colonne
    # sono specificate deve essere compatibile con quello che si aspetta il codice che esegue la
    # simulazione. Fare riferimento al commento all'inizio del file 'environment.py'
    CSV_COLUMNS = ["TemperaturaCelle", "PompaGlicoleMarcia", "TemperaturaMandataGlicole"]
    N_COLUMNS = 3

    # Path alle reti utilizzate per la simulazione
    TEMP_MODEL_ON_PATH = f"../../../modelli/temp_cell{CELL}_{TIME_UNIT}min__on"
    TEMP_MODEL_OFF_PATH = f"../../../modelli/temp_cell{CELL}_{TIME_UNIT}min__off"
    GLYCOL_MODEL_PATH = f"../../../modelli/glycol_cell{CELL}_{TIME_UNIT}min__off"

    # Directory in cui salavare la rete neurale che corrisponde all'agente
    OUTPUT_DIRECTORY = "../../../modelli/agente_rl/"

    # Ponendo questo parametro a <True>, si visualizza il comportamento dell'agente durante l'allenamento,
    # a costo di un tempo di esecuzione maggiore ed un'utilizzo di più memoria
    DEBUG = True

    REPLAY_BUFFER_SIZE = 5000

    # Learning rate
    LR = 0.001

    # Numero totale di episodi di allenamento
    TOTAL_EPISODES = 30
    # Per ogni episodio, l'agente sta nella simulazione per un'ora
    EPISODE_STEPS = 240
    # Dimensione di una batch di training
    BATCH_SIZE = 64

    # Fattore di ammortizzamento per le ricompense future
    GAMMA = 0.99
    # Parametro che regola l'update delle copie delle reti
    TAU = 0.05

    def __init__(self):
        self.N_COLUMNS = len(self.CSV_COLUMNS)
