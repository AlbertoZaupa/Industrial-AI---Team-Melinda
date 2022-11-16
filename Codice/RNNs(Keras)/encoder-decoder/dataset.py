import tensorflow as tf


def prepare_dataset_classification(df1, df2, n_input_features, past_window_size, forecast_window_size,
                                   batch_size, vectorized_labels=False):
    """
        df1: dataframe di input, tranne la variabile da classificare
        df2: dataframe contenente solo la variabile da classificare
        n_input_features: il numero di feature che rappresentano l'input al sistema (TemperaturaMandataGlicole)
        past_window_size: il numero di minuti per cui si guarda nel passato
        forecast_size: il numero di minuti per cui si prevede nel futuro
        batch_size: dimensione di una batch durante la fase di training
        vectorized_labels: se True le labels sono un vettore di lunghezza <forecast_window_size>, altrimenti le labels
            sono uno scalare corrispondente al valore della variabile al termine della finestra di predizione
      """

    total_size = past_window_size + forecast_window_size
    dataset2 = tf.one_hot(df2.values.astype('uint8'), 2)
    data = tf.concat([dataset2, df1.values.astype('float32')], axis=1)
    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    return split_inputs_and_outputs(data, forecast_window_size, n_input_features, batch_size, vectorized_labels,
                                    regression=False)


def prepare_dataset_regression(df, n_input_features, past_window_size, forecast_window_size, batch_size,
                               vectorized_labels=False):
    """
      df: dataframe di input
      n_input_features: il numero di feature che rappresentano l'input al sistema (TemperaturaMandataGlicole)
      past_window_size: il numero di minuti per cui si guarda nel passato
      forecast_size: il numero di minuti per cui si prevede nel futuro
      batch_size: dimensione di una batch durante la fase di training
      vectorized_labels: se True le labels sono un vettore di lunghezza <forecast_window_size>, altrimenti le labels
          sono uno scalare corrispondente al valore della variabile al termine della finestra di predizione
    """

    total_size = past_window_size + forecast_window_size
    data = tf.data.Dataset.from_tensor_slices(df.values.astype('float32'))
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    return split_inputs_and_outputs(data, forecast_window_size, n_input_features, batch_size, vectorized_labels)


def split_inputs_and_outputs(dataset, forecast_window_size, n_input_features, batch_size, vectorized_labels=False,
                             regression=True):
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
            return k[-labels_range:, 0]
        else:
            return tf.reshape(k[-labels_range:, :2], shape=(labels_range, 2))

    # k[:-forecast_size] è la matrice che contiene il valore delle variabili di stato
    # e delle variabili di input nei <past_window_size> minuti precedenti
    #
    # k[-forecast_size:, -n_input_features:] è la matrice che contiene il valore delle
    # variabili di input nei <forecast_size> minuti successivi
    #
    # k[-forecast_size:, 0] è il valore della variabile che vogliamo predirre
    # nei <forecast_size> minuti successivi

    dataset = dataset.map(lambda k: ((k[:-forecast_window_size],
                                      future_inputs_lambda(k)),
                                     labels_lambda(k)))

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
