import tensorflow as tf

GRU_UNITS = 16


# Un modello basato sull'architettura encoder-decoder
def encoder_decoder_arch(past_window_size, n_state_features, forecast_window_size, n_input_features,
                         regression=True, whole_output_sequence=False, n_labels=1):
    global GRU_UNITS

    # La parte della rete che ha il ruolo di encoder è una LSTM
    past_inputs = tf.keras.Input(shape=(past_window_size, n_state_features), name='past_inputs')
    encoder, state = tf.keras.layers.GRU(GRU_UNITS, return_state=True,
                                         dropout=0.1, recurrent_dropout=0.3)(past_inputs)

    # La parte della rete che ha il ruolo di decoder riceve in input la rappresentazione
    # dello 'stato' o 'contesto' del sistema che è stata prodotta dall'encoder, insieme
    # ai valori futuri della temperatura del glicole

    # Possiamo richiedere alla rete di prevedere l'intera sequenza di valori futuri nella finestra
    # temporale scelta, oppure di prevedere solo il valore al termine della finestra
    future_inputs_shape = (forecast_window_size, n_input_features) if whole_output_sequence else (n_input_features, forecast_window_size)
    future_inputs = tf.keras.Input(shape=future_inputs_shape, name='future_inputs')
    decoder = tf.keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(future_inputs, initial_state=state)
    gru = tf.keras.layers.GRU(GRU_UNITS * 4, return_sequences=True,
                              dropout=0.1, recurrent_dropout=0.3)(decoder)

    # Dei layer lineari vengono aggiunti al decoder
    x = tf.keras.layers.Dense(32, activation='relu')(gru)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    finalize_model = output_layer_codec_regression if regression else output_layer_codec_classification
    model = finalize_model(past_inputs, future_inputs, x, n_labels)
    return model


def seq2seq_arch(past_window_size, n_features, forecast_window_size, n_labels=1, regression=True):
    global GRU_UNITS

    inputs = tf.keras.Input(shape=(past_window_size, n_features), name='past_inputs')
    encoder, state = tf.keras.layers.GRU(GRU_UNITS, return_state=True,
                                         dropout=0.1, recurrent_dropout=0.3)(inputs)
    repeat_layer = tf.keras.layers.RepeatVector(forecast_window_size)(encoder)
    decoder = tf.keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(repeat_layer, state)
    gru = tf.keras.layers.GRU(GRU_UNITS * 4, return_sequences=True, dropout=0.1,
                              recurrent_dropout=0.3)(decoder)
    x = tf.keras.layers.Dense(32)(gru)
    finalize_model = output_layer_seq2seq_regression if regression else output_layer_seq2seq_classification
    return finalize_model(inputs, x, n_labels)


def output_layer_codec_regression(past_inputs, future_inputs, second_to_last_layer, n_labels=1):
    output = tf.keras.layers.Dense(n_labels)(second_to_last_layer)

    # Il modello viene compilato
    model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    return model


def output_layer_codec_classification(past_inputs, future_inputs, second_to_last_layer, n_labels=1):
    if n_labels == 1:
        output = tf.keras.layers.Dense(n_labels, activation='sigmoid')(second_to_last_layer)
        loss = "binary_crossentropy"
    else:
        output = tf.keras.layers.Dense(n_labels * 2, activation='softmax')(second_to_last_layer)
        loss = "categorical_crossentropy"

    # Il modello viene compilato
    model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)
    model.compile(loss=loss, optimizer='adam', metrics=["accuracy"])

    return model


def output_layer_seq2seq_regression(inputs, second_to_last_layer, n_labels=1):
    output = tf.keras.layers.Dense(n_labels)(second_to_last_layer)

    # Il modello viene compilato
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    return model


def output_layer_seq2seq_classification(inputs, second_to_last_layer, n_labels=1):
    if n_labels == 1:
        output = tf.keras.layers.Dense(n_labels, activation='sigmoid')(second_to_last_layer)
        loss = "binary_crossentropy"
    else:
        output = tf.keras.layers.Dense(n_labels * 2, activation='softmax')(second_to_last_layer)
        loss = "categorical_crossentropy"

    # Il modello viene compilato
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer='adam', metrics=["accuracy"])

    return model

