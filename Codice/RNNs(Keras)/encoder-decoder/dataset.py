import tensorflow as tf


def prepare_dataset(df, n_input_features, past_window_size, forecast_size, batch_size, shuffle=False):
    """
      df: dataframe di input
      n_input_features: il numero di feature che rappresentano l'input al sistema (TemperaturaMandataGlicole)
      past_window_size: il numero di minuti per cui si guarda nel passato
      forecast_size: il numero di minuti per cui si prevede nel futuro
      batch_size: dimensione di una batch durante la fase di training
    """

    total_size = past_window_size + forecast_size
    data = tf.data.Dataset.from_tensor_slices(df.values.astype('float32'))
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    if shuffle:
        shuffle_buffer_size = len(df)
        data = data.shuffle(shuffle_buffer_size)

    # k[:-forecast_size] è la matrice che contiene il valore delle variabili di stato
    # e delle variabili di input nei <past_window_size> minuti precedenti
    #
    # k[-forecast_size:, -n_input_features:] è la matrice che contiene il valore delle
    # variabili di input nei <forecast_size> minuti successivi
    #
    # k[-forecast_size:, 0] è il valore della variabile che vogliamo predirre
    # nei <forecast_size> minuti successivi

    data = data.map(lambda k: ((k[:-forecast_size],
                                tf.transpose(k[-forecast_size:, -n_input_features:])),
                               k[-1:, 0]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
