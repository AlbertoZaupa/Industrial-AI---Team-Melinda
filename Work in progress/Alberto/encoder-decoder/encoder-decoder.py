import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# La funzione riceve un dataframe pandas e organizza i dati in due tensori 3D. Il primo contiene
# i dati della cella passati, mentre il secondo contiene la temperatura del glicole futura

def prepare_dataset(df, n_input_features, past_window_size, forecast_size, batch_size):
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
                             k[-forecast_size:,-n_input_features:]),
                             k[-forecast_size:,0]))
  
  return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


  # Un modello basato sull'architettura encoder-decoder

def encoder_decoder(past_window_len, n_state_features, future_window_len, n_input_features):

  LSTM_UNITS = 16

  # La parte della rete che ha il ruolo di encoder è una LSTM

  past_inputs = keras.Input(shape=(past_window_len, n_state_features), name='past_inputs')
  encoder = keras.layers.LSTM(LSTM_UNITS, return_state=True)
  encoder_outputs, state_h, state_c = encoder(past_inputs)

  # La parte della rete che ha il ruolo di decoder riceve in input la rappresentazione
  # dello 'stato' o 'contesto' del sistema che è stata prodotta dall'encoder, insieme
  # ai valori futuri della temperatura del glicole

  future_inputs = keras.Input(shape=(future_window_len, n_input_features), name='future_inputs')
  decoder_lstm = keras.layers.LSTM(LSTM_UNITS, return_sequences=True)
  x = decoder_lstm(future_inputs, initial_state=[state_h, state_c])

  # Dei layer lineari vengono aggiunti al decoder

  x = keras.layers.Dense(16, activation='relu')(x)
  x = keras.layers.Dense(16, activation='relu')(x)
  output = keras.layers.Dense(1, activation='relu')(x)

  # Il modello viene compilato 
  model = keras.models.Model(inputs=[past_inputs,future_inputs], outputs=output)
  #optimizer = keras.optimizers.Adam()
  #loss = keras.losses.Huber()
  model.compile(loss='mae', optimizer='adam', metrics=["mae"])

  return model


# In questo blocco di codice vengono definite le feauture che ci interessano.
#
# Possiamo raggruppare le feature considerate in 3 categorie:
#   - La feature di cui vogliamo predirre il valore (TemperaturaCelle)
#   - Le feaure che rappresentano lo stato corrente del sistema (TemperaturaCelle, PompaGlicoleMarcia, TemperaturaRitornoGlicole, PercentualeAperturaValvolaMiscelatrice)
#   - Le feature che rappresentano l'input esterno al sistema (TemperaturaMandataGlicole)
#
# Dopo aver estratto dal dataframe le feature che sono d'interesse, vengono preparati
# i dataset per l'allenamento, la validazione ed il test della rete neurale


columns = [
    "TemperaturaCelle",
    "PompaGlicoleMarcia",
    "TemperaturaRitornoGlicole",
    "PercentualeAperturaValvolaMiscelatrice",
    "TemperaturaMandataGlicole",
]
prefix = "Ipogeo2_Cella13"
columns = [prefix + column for column in columns]
n_columns = len(columns)
target_column = columns[0]


df = pd.read_csv(f"01 09 - 04 10/CSV_celle/{prefix}.csv")
df = df[columns]
df = df.replace({',': '.'}, regex=True).astype(float)
df = df.fillna(0)


df_len = df.shape[0]
TRAIN_SPLIT = int(0.7 * df_len)
VALIDATION_SPLIT = int(0.2 * df_len)


train_df = df.iloc[:TRAIN_SPLIT]
val_df = df.iloc[TRAIN_SPLIT:TRAIN_SPLIT + VALIDATION_SPLIT]
test_df = df.iloc[TRAIN_SPLIT + VALIDATION_SPLIT:]


N_INPUT_FEATURES = 1
PAST_WINDOW_SIZE = 6*60
FORECAST_SIZE = 60
BATCH_SIZE = 72
train_data = prepare_dataset(train_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
validation_data = prepare_dataset(val_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
test_data = prepare_dataset(test_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)


# In questo blocco di codice viene compilata ed allenata la rete neurale

model = encoder_decoder(PAST_WINDOW_SIZE, len(columns), FORECAST_SIZE, N_INPUT_FEATURES)

history = model.fit(train_data, epochs=10, validation_data=validation_data, verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# Vengono quindi mostrati i risultati della rete sul dataset di test

scaling_factor = df[target_column].max()
for i, data in enumerate(test_data.take(1)):
  (past, future), truth = data
  truth = truth * scaling_factor
  pred = model.predict((past,future)) * scaling_factor
  plt.plot(pred.flatten(), label='Prediction')
  plt.plot(truth.numpy().flatten(),label='Truth')
  plt.legend()
  plt.show()

model.evaluate(test_data)