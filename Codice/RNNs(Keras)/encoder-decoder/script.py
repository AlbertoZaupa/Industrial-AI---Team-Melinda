import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import prepare_dataset
from model import encoder_decoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cella", type=int, required=True,
                        help="Cella su cui allenare il modello")
    parser.add_argument("-p", "--past-window", type=int, required=False, default=12,
                        help="Finestra temporale (in h) su cui allenare il modello")
    parser.add_argument("-f", "--forecast-window", type=int, required=False, default=1,
                        help="Finestra temporale (in h) su cui fare la previsione")

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
    # columns = [prefix + column for column in columns]
    n_columns = len(columns)
    target_column = columns[0]

    df = pd.read_csv(f"../../../CSV/october/Cella_{n_cella}.csv")
    df = df[columns]
    df = df.replace({',': '.'}, regex=True).astype(float)

    # per ogni colonna sostituisce i NA con la media locale (calcolata senza NA - skipna=True)
    for column in df.columns:
        df[column] = df[column].fillna((df[column].mean(skipna=True)))

    # Scaling nel range [0, 1]
    scaling_factors = {column: (df[column].min(), df[column].max()) for column in df.columns}
    scaled_df = pd.DataFrame()
    for column in df.columns:
        scaled_df[column] = (df[column] - scaling_factors[column][0]) / (
                scaling_factors[column][1] - scaling_factors[column][0])

    # OPPURE, per scalare

    from sklearn.preprocessing import MinMaxScaler

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

    # viene plottato il valore della variabile target
    x_axis = range(test_df.shape[0])

    # plt.plot(x_axis, test_df[target_column] * scaling_factors[target_column][1])
    # plt.show()

    # vengono preparati i dataset
    N_INPUT_FEATURES = 1
    # PAST_WINDOW_SIZE = 12*60
    PAST_WINDOW_SIZE = args.past_window * 60
    # FORECAST_SIZE = 1*60
    FORECAST_SIZE = args.forecast_window * 60
    BATCH_SIZE = 256
    train_data = prepare_dataset(train_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
    validation_data = prepare_dataset(val_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
    test_data = prepare_dataset(test_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)

    # la rete neurale
    model = encoder_decoder(PAST_WINDOW_SIZE, len(columns), FORECAST_SIZE, N_INPUT_FEATURES)

    # la rete neurale viene allenata sul dataset di training
    history = model.fit(train_data, epochs=25, validation_data=validation_data, verbose=2, shuffle=False)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # viene plottato il confronto tra i valori predetti dalla rete neurale ed i valori reali
    scaling_factor = scaling_factors[target_column][1]

    truth_flat = []
    pred_flat = []
    for batch in test_data:
        (past, future), truth = batch
        truth = truth * scaling_factor
        pred = model.predict((past, future)) * scaling_factor
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
