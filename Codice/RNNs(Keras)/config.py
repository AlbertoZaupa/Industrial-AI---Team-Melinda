class Config:
    # Numero di campioni passati che vengono dati in input alle reti neruali
    PAST_WINDOW_SIZE = 360
    # La rete neruale effettua una previsione per i <FORECAST_WINDOW_SIZE> minuti nel futuro
    FORECAST_WINDOW_SIZE = 5

    # iperparametri dei modelli
    BATCH_SIZE = 256
    EPOCHS = 25

    # Qui vanno indicati i percori al dataset e la directory in cui si desidera salvare la rete neurale
    DATASET_PATH = "../../../datasets/Cella_13.csv"
    OUTPUT_PATH = "/dev/null"

    # Colonne del dataset considerate
    STATE_COLUMNS = [
        "TemperaturaCelle",
        "PercentualeAperturaValvolaMiscelatrice",
    ]
    CONTROL_COLUMNS = [
        "TemperaturaMandataGlicole",
    ]
