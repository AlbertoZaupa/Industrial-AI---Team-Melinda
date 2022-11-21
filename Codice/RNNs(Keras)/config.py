class Config:
    # Numero di campioni passati che vengono dati in input alle reti neruali
    PAST_WINDOW_SIZE = 360

    # La rete neruale effettua una previsione per i <FORECAST_WINDOW_SIZE> minuti nel futuro
    FORECAST_WINDOW_SIZE = 5

    # Cella considerata
    CELL = 13

    BATCH_SIZE = 256

    # Qui vanno indicati i percori al dataset e la directory in cui si desidera salvare la rete neurale
    DATASET_PATH = "..."
    OUTPUT_PATH = "..."

    # Colonne del dataset considerate
    COLUMNS = [
        # LA PRIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI VIENE PREDETTO IL VALORE
        # L'ULTIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI IL MODELLO RICEVE I VALORI FUTURI IN INPUT
        "PompaGlicoleMarcia",
        "TemperaturaCelle",
        "TemperaturaMandataGlicole",
    ]

    # Numero di colonne di cui la rete neurale riceve in input i valori passati
    N_STATE_COLUMNS = 3

    # Numero di colonne di cui la rete neurale riceve in input i valori futuri fino a <FORECAST_WINDOW_SIZE>
    N_INPUT_FEATURES = 0

    def __init__(self):
        self.N_STATE_COLUMNS = len(self.COLUMNS)
