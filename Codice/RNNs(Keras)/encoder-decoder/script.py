import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dataset import prepare_dataset
from model import encoder_decoder_arch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cella", type=int, required=True,
                        help="Cella su cui allenare il modello")
    parser.add_argument("-p", "--past-window", type=int, required=False, default=12,
                        help="Finestra temporale (in h) su cui allenare il modello")
    parser.add_argument("-f", "--forecast-window", type=int, required=False, default=1,
                        help="Finestra temporale (in h) su cui fare la previsione")
    parser.add_argument("-ds", "--dataset-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)

    args = parser.parse_args()

    # Le colonne che vengono utilizzate durante il training
    # TODO sostituire 'Temperatura mandata glicole' con 'Temperatura glicole nominale'
    columns = [
        # LA PRIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI VIENE PREDETTO IL VALORE
        "TemperaturaRitornoGlicole",
        "TemperaturaCelle",
        "PompaGlicoleMarcia",
        "PercentualeAperturaValvolaMiscelatrice",
        # L'ULTIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI IL MODELLO RICEVE I VALORI FUTURI IN INPUT
        # "TemperaturaMandataGlicole",
        "TemperaturaMandataGlicoleNominale"
    ]
    n_cella = args.cella
    n_columns = len(columns)
    target_column = columns[0]
    dataset_path = args.dataset_path
    output_path = args.output_path

    df = pd.read_csv(f"{dataset_path}/{n_cella}.csv")
    df = df[columns]
    df = df.replace({',': '.'}, regex=True).astype(float)

    # per ogni colonna sostituisce i NA con la media locale (calcolata senza NA - skipna=True)
    for column in df.columns:
        df[column] = df[column].fillna((df[column].mean(skipna=True)))

    # Fattori di scala
    scaling_factors = {}
    for column in df.columns:
        min = df[column].min()
        max = df[column].max()
        scaling_factors[column] = {"min": min, "scale": max - min}

    # Il dataset viene riscalato
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df), columns=df.columns
    )

    # training, validation e test split
    df_len = df.shape[0]
    TRAIN_SPLIT = int(0.7 * df_len)
    VALIDATION_SPLIT = int(0.2 * df_len)

    train_df = scaled_df.iloc[:TRAIN_SPLIT]
    val_df = scaled_df.iloc[TRAIN_SPLIT:TRAIN_SPLIT + VALIDATION_SPLIT]
    test_df = scaled_df.iloc[TRAIN_SPLIT + VALIDATION_SPLIT:]

    # vengono preparati i dataset
    N_INPUT_FEATURES = 1
    PAST_WINDOW_SIZE = args.past_window * 60
    FORECAST_SIZE = args.forecast_window * 60
    BATCH_SIZE = 256
    train_data = prepare_dataset(train_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
    validation_data = prepare_dataset(val_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
    test_data = prepare_dataset(test_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)

    # la rete neurale
    model = encoder_decoder_arch(PAST_WINDOW_SIZE, len(columns), FORECAST_SIZE, N_INPUT_FEATURES)

    # la rete neurale viene allenata sul dataset di training
    history = model.fit(train_data, epochs=25, validation_data=validation_data, verbose=2, shuffle=False)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # viene plottato il confronto tra i valori predetti dalla rete neurale ed i valori reali
    min = scaling_factors[target_column]["min"]
    scale = scaling_factors[target_column]["scale"]
    truth_flat = []
    pred_flat = []
    for batch in test_data:
        (past, future), truth = batch
        # I valori di ground_truth e prediction vengono riportati nel range originario
        truth = truth * scale + min
        pred = model.predict((past, future)) * scale + min
        for v in truth[:, -1]:
            truth_flat.append(v)
        for v in pred[:, -1]:
            pred_flat.append(v)

    mse = ((np.array([t.numpy() for t in truth_flat]) - np.array([p[0] for p in pred_flat])) ** 2).mean()

    plt.title(f'Predictions vs. ground truth')
    plt.suptitle(f'MSE: %.4f' % mse)
    plt.plot(truth_flat, label="Truth")
    plt.plot(pred_flat, label="Prediction")
    plt.legend()
    plt.show()

    # Il modello viene salvato
    model.save(f"{output_path}")
