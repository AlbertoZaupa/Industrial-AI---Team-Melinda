import keras

LSTM_UNITS = 16

# Un modello basato sull'architettura encoder-decoder

def encoder_decoder(past_window_len, n_state_features, future_window_len, n_input_features):
    global LSTM_UNITS

    # La parte della rete che ha il ruolo di encoder è una LSTM

    past_inputs = keras.Input(shape=(past_window_len, n_state_features), name='past_inputs')
    encoder_outputs, state_h, state_c = keras.layers.LSTM(LSTM_UNITS, return_state=True)(past_inputs)

    # La parte della rete che ha il ruolo di decoder riceve in input la rappresentazione
    # dello 'stato' o 'contesto' del sistema che è stata prodotta dall'encoder, insieme
    # ai valori futuri della temperatura del glicole

    future_inputs = keras.Input(shape=(n_input_features, future_window_len), name='future_inputs')
    decoder_lstm = keras.layers.LSTM(LSTM_UNITS, return_sequences=True)(future_inputs, initial_state=[state_h, state_c])

    # Dei layer lineari vengono aggiunti al decoder

    x = keras.layers.Dense(16, activation='relu')(decoder_lstm)
    x = keras.layers.Dense(16, activation='relu')(x)
    output = keras.layers.Dense(1, activation='relu')(x)

    # Il modello viene compilato 
    model = keras.models.Model(inputs=[past_inputs,future_inputs], outputs=output)
    print(model.summary())
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])

    return model