class Config:
    PAST_WINDOW_SIZE = 360
    FORECAST_WINDOW_SIZE = 5
    CELL = 13
    N_STATE_FEATURES = 3
    N_INPUT_FEATURES = 0
    BATCH_SIZE = 256
    DATASET_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets"
    OUTPUT_PATH = f"/Users/albertozaupa/Desktop/Projects/Melinda/modelli/glycol_cell{CELL}_{FORECAST_WINDOW_SIZE}min__off"
    COLUMNS = [
        # LA PRIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI VIENE PREDETTO IL VALORE
        # L'ULTIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI IL MODELLO RICEVE I VALORI FUTURI IN INPUT
        "TemperaturaMandataGlicole",
        "TemperaturaCelle",
        "PompaGlicoleMarcia",
    ]

    def __init__(self):
        self.N_STATE_FEATURES = len(self.COLUMNS)
