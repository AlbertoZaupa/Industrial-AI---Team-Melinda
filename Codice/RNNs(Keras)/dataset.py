import pandas as pd
import tensorflow as tf
import numpy as np


def prepare_dataset_classification(df1, df2, n_input_features, past_window_size, forecast_window_size,
                                   n_labels, batch_size, vectorized_labels=False):
    """
        A1: dataframe/nd-array di input, tranne la variabile da classificare
        A2: dataframe/nd-array contenente solo la variabile da classificare
        n_input_features: il numero di feature che rappresentano l'input al sistema (TemperaturaMandataGlicole)
        past_window_size: il numero di minuti per cui si guarda nel passato
        forecast_size: il numero di minuti per cui si prevede nel futuro
        BATCH_SIZE: dimensione di una batch durante la fase di training
        vectorized_labels: se True le labels sono un vettore di lunghezza <forecast_window_size>, altrimenti le labels
            sono uno scalare corrispondente al valore della variabile al termine della finestra di predizione
      """

    data_numpy1 = df1.values.astype('float32') if type(df1) == pd.DataFrame or type(df1) == pd.Series else df1
    data_numpy2 = df2.values.astype('uint8') if type(df2) == pd.DataFrame or type(df2) == pd.Series else df2
    total_size = past_window_size + forecast_window_size
    data_numpy2 = tf.reshape(tf.one_hot(data_numpy2, 2), (data_numpy2.shape[0], 2))  # [OFF_COLUMN, ON_COLUMN]
    data_numpy = np.concatenate((data_numpy2, data_numpy1), axis=1)
    data = tf.data.Dataset.from_tensor_slices(data_numpy)
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    return split_inputs_and_outputs(data, forecast_window_size, n_input_features, n_labels, batch_size,
                                    vectorized_labels, regression=False)


def prepare_dataset_regression(df, n_input_features, past_window_size, forecast_window_size, n_labels,
                               batch_size, vectorized_labels=False):
    """
      A: dataframe/nd-array di input
      n_input_features: il numero di feature che rappresentano l'input al sistema (TemperaturaMandataGlicole)
      past_window_size: il numero di minuti per cui si guarda nel passato
      forecast_size: il numero di minuti per cui si prevede nel futuro
      BATCH_SIZE: dimensione di una batch durante la fase di training
      vectorized_labels: se True le labels sono un vettore di lunghezza <forecast_window_size>, altrimenti le labels
          sono uno scalare corrispondente al valore della variabile al termine della finestra di predizione
    """

    data_numpy = df.values.astype('float32') if type(df) == pd.DataFrame or type(df) == pd.Series else df
    total_size = past_window_size + forecast_window_size
    data = tf.data.Dataset.from_tensor_slices(data_numpy)
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    return split_inputs_and_outputs(data, forecast_window_size, n_input_features, n_labels, batch_size,
                                    vectorized_labels)


def split_inputs_and_outputs(dataset, forecast_window_size, n_input_features, n_labels, batch_size,
                             vectorized_labels=False, regression=True):
    # Per la feature che cerchiamo di predirre, possiamo scegliere di ottenere come
    # label il vettore di valori futuri dal momento corrente
    # al termine di <forecast_size>, oppure solo l'ultimo elemento del vettore.

    labels_range = forecast_window_size if vectorized_labels else 1

    def future_inputs_lambda(k):
        if vectorized_labels:
            return k[-forecast_window_size:, -n_input_features:]
        else:
            return tf.transpose(k[-forecast_window_size:, -n_input_features:])

    def labels_lambda(k):
        if regression:
            return k[-labels_range:, :n_labels]
        else:
            return tf.reshape(k[-labels_range:, :2 * n_labels], shape=(labels_range, 2 * n_labels))

    # k[:-forecast_size] è la matrice che contiene il valore delle variabili di stato
    # e delle variabili di input nei <past_window_size> minuti precedenti
    #
    # k[-forecast_size:, -n_input_features:] è la matrice che contiene il valore delle
    # variabili di input nei <forecast_size> minuti successivi
    #
    # k[-forecast_size:, 0] è il valore della variabile che vogliamo predirre
    # nei <forecast_size> minuti successivi

    if n_input_features > 0:
        dataset = dataset.map(lambda k: ((k[:-forecast_window_size],
                                         future_inputs_lambda(k)),
                                         labels_lambda(k)))
    else:
        dataset = dataset.map(lambda k: (k[:-forecast_window_size],
                                         labels_lambda(k)))

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def one_hot_encoding(A, low_end, high_end, step=1):
    """
    A: vettore colonna di dimensione (n,) oppure (n, 1)
    low_end: valore minimo di A
    high_end: valore massimo di A
    step: fattore di quantizzazione

    Output: matrice di dimensioni (n, (high_end - low_end) / step )
    """
    assert high_end > low_end

    output = np.zeros((A.shape[0], int((high_end - low_end) / step)), dtype=np.uint8)

    for i in range(A.shape[0]):
        idx = int((A[i] - low_end) / step)
        output[i, idx] = 1

    assert output.dtype == np.uint8
    return output
