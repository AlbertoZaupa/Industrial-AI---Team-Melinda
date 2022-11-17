import tensorflow as tf

LSTM_UNITS = 16


# Un modello basato sull'architettura encoder-decoder
def encoder_decoder_arch(past_window_len, n_state_features, future_window_len, n_input_features,
                         regression=True, whole_output_sequence=False, n_labels=1):
    global LSTM_UNITS

    # La parte della rete che ha il ruolo di encoder è una LSTM
    past_inputs = tf.keras.Input(shape=(past_window_len, n_state_features), name='past_inputs')
    encoder, state = tf.keras.layers.GRU(LSTM_UNITS, return_state=True,
                                         dropout=0.1, recurrent_dropout=0.3)(past_inputs)

    # La parte della rete che ha il ruolo di decoder riceve in input la rappresentazione
    # dello 'stato' o 'contesto' del sistema che è stata prodotta dall'encoder, insieme
    # ai valori futuri della temperatura del glicole

    # Possiamo richiedere alla rete di prevedere l'intera sequenza di valori futuri nella finestra
    # temporale scelta, oppure di prevedere solo il valore al termine della finestra
    future_inputs_shape = (future_window_len, n_input_features) if whole_output_sequence else (n_input_features, future_window_len)
    future_inputs = tf.keras.Input(shape=future_inputs_shape, name='future_inputs')
    decoder = tf.keras.layers.GRU(LSTM_UNITS, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(future_inputs, initial_state=state)
    gru = tf.keras.layers.GRU(LSTM_UNITS*4, return_sequences=True,
                              dropout=0.1, recurrent_dropout=0.3)(decoder)

    # Dei layer lineari vengono aggiunti al decoder
    x = tf.keras.layers.Dense(32, activation='relu')(gru)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    finalize_model = finalize_model_regression if regression else finalize_model_classification
    model = finalize_model(past_inputs, future_inputs, x, n_labels)
    print(model.summary())
    return model


def finalize_model_regression(past_inputs, future_inputs, second_to_last_layer, n_labels=1):
    output = tf.keras.layers.Dense(n_labels)(second_to_last_layer)

    # Il modello viene compilato
    model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    return model


def finalize_model_classification(past_inputs, future_inputs, second_to_last_layer, n_labels=1):
    output = tf.keras.layers.Dense(n_labels * 2, activation='softmax')(second_to_last_layer)

    # Il modello viene compilato
    model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model


def seq2seq_arch(past_window_len, n_features, future_window_len, n_labels=1):
    global LSTM_UNITS

    inputs = tf.keras.Input(shape=(past_window_len, n_features), name='past_inputs')
    encoder, state = tf.keras.layers.GRU(LSTM_UNITS, return_state=True,
                                         dropout=0.1, recurrent_dropout=0.3)(inputs)
    repeat_layer = tf.keras.layers.RepeatVector(future_window_len)(encoder)
    decoder = tf.keras.layers.GRU(LSTM_UNITS, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(repeat_layer, state)
    gru = tf.keras.layers.GRU(LSTM_UNITS*4, return_sequences=True, dropout=0.1,
                              recurrent_dropout=0.3)(decoder)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32))(gru)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_labels))(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    return model

