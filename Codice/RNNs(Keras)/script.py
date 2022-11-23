import matplotlib.pyplot as plt
from dataset import *
from model import *
from config import Config
Config()


def seq2seq_example():
    df = pd.read_csv(Config.DATASET_PATH)
    df = df.loc[:, ~df.columns.isin(["Date"])]
    df = df.replace({',': '.'}, regex=True).astype(float)
    # per ogni colonna sostituisce i NA con la media locale (calcolata senza NA - skipna=True)
    for column in df.columns:
        df[column] = df[column].fillna((df[column].mean(skipna=True)))

    # training, validation e test split
    df_len = df.shape[0]
    train_split = int(0.7 * df_len)
    val_split = int(0.2 * df_len)

    # i dataset di training, validation e test
    train_data, validation_data, test_data = prepare_dataset(df=df, training_split=train_split,
                                                             validation_split=val_split,
                                                             state_features=Config.STATE_COLUMNS,
                                                             control_features=[],  # l'architettura non distingue tra
                                                             # variabili di stato e di controllo
                                                             past_window_size=Config.PAST_WINDOW_SIZE,
                                                             forecast_window_size=Config.FORECAST_WINDOW_SIZE,
                                                             batch_size=Config.BATCH_SIZE, extended_output=True)

    # la rete neurale
    model = seq2seq(past_window_size=Config.PAST_WINDOW_SIZE, n_features=len(Config.STATE_COLUMNS),
                    forecast_window_size=Config.FORECAST_WINDOW_SIZE)

    # allenamento
    train_nn(model, train_data, validation_data, test_data)


def encoder_decoder_example():
    df = pd.read_csv(Config.DATASET_PATH)
    df = df.loc[:,  ~df.columns.isin(["Date"])]
    df = df.replace({',': '.'}, regex=True).astype(float)
    # per ogni colonna sostituisce i NA con la media locale (calcolata senza NA - skipna=True)
    for column in df.columns:
        df[column] = df[column].fillna((df[column].mean(skipna=True)))

    # training, validation e test split
    df_len = df.shape[0]
    train_split = int(0.7 * df_len)
    val_split = int(0.2 * df_len)

    # i dataset di training, validation e test
    train_data, validation_data, test_data = prepare_dataset(df=df, training_split=train_split,
                                                             validation_split=val_split,
                                                             state_features=Config.STATE_COLUMNS,
                                                             control_features=Config.CONTROL_COLUMNS,
                                                             past_window_size=Config.PAST_WINDOW_SIZE,
                                                             forecast_window_size=Config.FORECAST_WINDOW_SIZE,
                                                             batch_size=Config.BATCH_SIZE, extended_output=True)
    n_state_features = len(Config.STATE_COLUMNS) + len(Config.CONTROL_COLUMNS)
    n_control_features = len(Config.CONTROL_COLUMNS)

    # la rete neurale
    model = encoder_decoder(past_window_size=Config.PAST_WINDOW_SIZE, forecast_window_size=Config.FORECAST_WINDOW_SIZE,
                            n_state_features=n_state_features, n_control_features=n_control_features,
                            extended_output=True)

    # allenamento
    train_nn(model, train_data, validation_data, test_data)


def train_nn(nn, train_data, validation_data, test_data):
    history = nn.fit(train_data, epochs=Config.EPOCHS, validation_data=validation_data, verbose=2, shuffle=True)

    # confronto training loss - validation loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # viene visualizzato il confronto tra i valori predetti dalla rete neurale ed i valori reali
    truth_flat = [[] for i in range(Config.FORECAST_WINDOW_SIZE)]
    pred_flat = [[] for i in range(Config.FORECAST_WINDOW_SIZE)]
    for batch in test_data:
        _input, truth = batch
        pred = nn.predict(_input)
        for v in truth[:]:
            for i in range(Config.FORECAST_WINDOW_SIZE):
                truth_flat[i].append(v[i])
        for v in pred[:]:
            for i in range(Config.FORECAST_WINDOW_SIZE):
                pred_flat[i].append(v[i])

    for i in range(Config.FORECAST_WINDOW_SIZE):
        truth = truth_flat[i]
        pred = pred_flat[i]
        plt.plot(truth, label="Truth")
        plt.plot(pred, label="Prediction")
        plt.legend()
        plt.show()

    # il modello viene salvato
    if input("Save network parameters? (Y to save) ") == "U":
        nn.save(Config.OUTPUT_PATH)


if __name__ == '__main__':
    encoder_decoder_example()  # seq2seq_example()
