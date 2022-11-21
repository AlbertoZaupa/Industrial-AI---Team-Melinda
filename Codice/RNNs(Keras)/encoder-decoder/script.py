import matplotlib.pyplot as plt
from dataset import *
from model import *
from config import Config
Config()


def load_dataset():
    df = pd.read_csv(f"{Config.DATASET_PATH}/Cella_{Config.CELL}_SH_merged.csv")
    df = df[Config.COLUMNS]
    df = df.replace({',': '.'}, regex=True).astype(float)

    # per ogni colonna sostituisce i NA con la media locale (calcolata senza NA - skipna=True)
    for column in df.columns:
        df[column] = df[column].fillna((df[column].mean(skipna=True)))

    # training, validation e test split
    df_len = df.shape[0]
    train_split = int(0.7 * df_len)
    val_split = int(0.2 * df_len)

    training_ds = df.iloc[:train_split]
    validation_ds = df.iloc[train_split: val_split + train_split]
    test_ds = df.iloc[train_split + val_split:]

    train = prepare_dataset_regression(training_ds, Config.N_INPUT_FEATURES, Config.PAST_WINDOW_SIZE,
                                       Config.FORECAST_WINDOW_SIZE, n_labels=1, batch_size=Config.BATCH_SIZE,
                                       vectorized_labels=True)
    validation = prepare_dataset_regression(validation_ds, Config.N_INPUT_FEATURES,
                                            Config.PAST_WINDOW_SIZE, Config.FORECAST_WINDOW_SIZE,
                                            n_labels=1, batch_size=Config.BATCH_SIZE, vectorized_labels=True)
    test = prepare_dataset_regression(test_ds, Config.N_INPUT_FEATURES, Config.PAST_WINDOW_SIZE,
                                      Config.FORECAST_WINDOW_SIZE, n_labels=1,
                                      batch_size=Config.BATCH_SIZE, vectorized_labels=True)

    return train, validation, test


if __name__ == '__main__':
    train_data, validation_data, test_data = load_dataset()
    # la rete neurale
    model = seq2seq_arch(Config.PAST_WINDOW_SIZE, Config.N_STATE_FEATURES, Config.FORECAST_WINDOW_SIZE,
                         n_labels=1, regression=True)
    # model = encoder_decoder_arch(Config.PAST_WINDOW_SIZE, Config.N_STATE_FEATURES, Config.FORECAST_WINDOW_SIZE,
    #                              n_input_features=1, regression=False, whole_output_sequence=True, n_labels=1)
    # la rete neurale viene allenata sul dataset di training
    history = model.fit(train_data, epochs=25, validation_data=validation_data, verbose=2, shuffle=True)
    # Il modello viene salvato
    model.save(f"{Config.OUTPUT_PATH}")

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # viene plottato il confronto tra i valori predetti dalla rete neurale ed i valori reali
    truth_flat = [[] for i in range(Config.FORECAST_WINDOW_SIZE)]
    pred_flat = [[] for i in range(Config.FORECAST_WINDOW_SIZE)]
    for batch in test_data:
        _input, truth = batch
        pred = model.predict(_input)
        for v in truth[:]:
            for i in range(Config.FORECAST_WINDOW_SIZE):
                truth_flat[i].append(v[i])
            # truth_flat.append(0 if v[FORECAST_WINDOW_SIZE - 1][0] == 1 else 1)
        for v in pred[:]:
            for i in range(Config.FORECAST_WINDOW_SIZE):
                pred_flat[i].append(v[i])
            # pred_flat.append(0 if v[FORECAST_WINDOW_SIZE - 1][0] > 0.5 else 1)

    for i in range(Config.FORECAST_WINDOW_SIZE):
        truth = truth_flat[i]
        pred = pred_flat[i]
        plt.plot(truth, color="green")
        plt.plot(pred, color="red")
        plt.show()
    # plt.plot(truth_flat, label="Truth")
    # plt.plot(pred_flat, label="Prediction")
    # plt.legend()
    # plt.show()
