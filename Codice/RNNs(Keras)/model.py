import tensorflow as tf

GRU_UNITS = 16


def encoder_decoder(past_window_size,  # numero di campioni passati ricevuti in input
                 n_state_features,  # numero di variabili corrispondenti allo stato del sistema
                 forecast_window_size,  # orizzonte temporale della predizione (in minuti)
                 n_control_features,  # numero di variabili di controllo, di cui assumiamo di conoscere il valore futuro
                 output_as_sequence,  # se <True>, la previsione include ogni minuto a partire dal momento corrente
                                      # fino a <forecast_window_size>
                 ):
    # La parte della rete che ha il ruolo di encoder è una rete GRU
    past_inputs = tf.keras.Input(shape=(past_window_size, n_state_features), name='past_inputs')
    encoder, state = tf.keras.layers.GRU(GRU_UNITS, return_state=True,
                                         dropout=0.1, recurrent_dropout=0.3)(past_inputs)

    # La parte della rete che ha il ruolo di decoder riceve in input la rappresentazione
    # dello 'stato' o 'contesto' del sistema che è stata prodotta dall'encoder, insieme
    # ai valori futuri delle variabili di controllo

    # Possiamo richiedere alla rete di prevedere l'intera sequenza di valori futuri nella finestra
    # temporale scelta, oppure di prevedere solo il valore al termine della finestra
    future_inputs_shape = (forecast_window_size, n_control_features) if output_as_sequence else (
        n_control_features, forecast_window_size)
    future_inputs = tf.keras.Input(shape=future_inputs_shape, name='future_inputs')
    decoder = tf.keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(future_inputs, initial_state=state)
    gru = tf.keras.layers.GRU(GRU_UNITS * 4, return_sequences=True,
                              dropout=0.1, recurrent_dropout=0.3)(decoder)

    # Dei layer lineari vengono aggiunti al decoder
    x = tf.keras.layers.Dense(32, activation='relu')(gru)
    x = tf.keras.layers.Dense(16, activation='relu')(x)

    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics='loss')
    return model


def seq2seq(past_window_size,  # numero di campioni passati ricevuti in input
            n_features,  # numero di features passate in input alla rete
            forecast_window_size  # orizzonte temporale della predizione (in minuti)
            ):
    inputs = tf.keras.Input(shape=(past_window_size, n_features), name='past_inputs')
    encoder, state = tf.keras.layers.GRU(GRU_UNITS, return_state=True,
                                         dropout=0.1, recurrent_dropout=0.3)(inputs)
    repeat_layer = tf.keras.layers.RepeatVector(forecast_window_size)(encoder)
    decoder = tf.keras.layers.GRU(GRU_UNITS, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(repeat_layer, state)
    gru = tf.keras.layers.GRU(GRU_UNITS * 4, return_sequences=True, dropout=0.1,
                              recurrent_dropout=0.3)(decoder)
    x = tf.keras.layers.Dense(32)(gru)

    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics='loss')
    return model

