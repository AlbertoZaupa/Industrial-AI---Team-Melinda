import pandas as pd
import tensorflow as tf


def prepare_dataset(df: pd.DataFrame,  # dataframe pandas contenente il dataset
                    training_split: float,  # percentuale del dataset utilizzata per il training
                    validation_split: float,  # percentuale del dataset utilizzata per la validazion,
                    # la percentuale rimanente del dataset è utilizzata per la fase di test
                    state_features: list[str],  # lista delle colonne relative alle feature che compongono lo stato
                    # della cella il primo elemento di <state_features> corrisponde alla colonna target
                    control_features: list[str],  # lista delle colonne relative alle variabili di controllo
                    past_window_size: int,  # numero di campioni passati
                    forecast_window_size: int,  # orizzonte temporale delle predizioni del modello
                    batch_size: int,
                    extended_output: bool  # se <True>, le label includono tutta la finestra temporale dal momento
                    # presente fino a <forecast_window_size>, altrimenti solo l'unico elemento della finestra
                    ):
    state_df = df[state_features]
    control_df = df[control_features]
    combined_df = pd.concat((state_df, control_df), axis=1)

    # I dataset vengono divisi in training, validation e test
    df_training = combined_df.iloc[:training_split]
    df_validation = combined_df.iloc[training_split:training_split + validation_split]
    df_test = combined_df.iloc[training_split + validation_split:]

    n_control_features = len(control_features)
    return (pandas_df_to_tensorflow_ds(df_training, n_control_features, past_window_size, forecast_window_size,
                                       batch_size, extended_output),
            pandas_df_to_tensorflow_ds(df_validation, n_control_features, past_window_size, forecast_window_size,
                                       batch_size, extended_output),
            pandas_df_to_tensorflow_ds(df_test, n_control_features, past_window_size, forecast_window_size,
                                       batch_size, extended_output))


def pandas_df_to_tensorflow_ds(df: pd.DataFrame, n_control_features: int,
                               past_window_size: int, forecast_window_size: int, batch_size: int,
                               extended_output: bool):
    window_size = past_window_size + forecast_window_size
    dataset = tf.data.Dataset.from_tensor_slices(df)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda k: k.batch(window_size))

    labels_range = forecast_window_size if extended_output else 1

    if n_control_features > 0:
        dataset = dataset.map(lambda k: ((k[:-forecast_window_size],
                                          k[-forecast_window_size:, -n_control_features:]),
                                         k[-labels_range:, :1]))
    else:
        dataset = dataset.map(lambda k: (k[:-forecast_window_size],
                                         k[-labels_range:, :1]))

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
