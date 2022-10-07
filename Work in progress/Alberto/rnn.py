from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import isnan


# Alcune costanti

columns = [
    "PompaGlicoleMarcia",
    "TemperaturaCelle",
    "TemperaturaRitornoGlicole",
    "TemperaturaMandataGlicole",
    "PercentualeAperturaValvolaMiscelatrice"
]
prefix = "Ipogeo2_Cella13"
columns = [prefix + column for column in columns]
n_columns = len(columns)
target_column = prefix + "TemperaturaCelle"

TRAIN_PERCENTAGE = 0.75


# Preparazione dei valori di input ed output che vengono passati al modello

def prepare_dataset(ds, window_size = 60):
  """
  Da fare:
    - sostituire 'fillna(0)' con la media dei valori intorno
  """

  # vengono selezionate solamente le colonne specificate
  ds = ds[columns]
  ds = ds.replace({',': '.'}, regex=True).astype(float)
  ds = ds.fillna(0)

  # poichè il modello deve ricevere in input lo stato delle celle fino a <window_size> minuti nel passato
  # e la temperatura futura fino a <window_size> minuti nel futuro, il numero di campioni che possiamo
  # considerare è il seguente:
  n_samples = ds.shape[0] - 2 * window_size

  # vengono estratte le label
  Y = ds[target_column][2 * window_size:]
  Y = Y.values.reshape(Y.shape[0], 1)
  assert Y.shape == (n_samples, 1), str(Y.shape)


  # la seguente sezione di codice inizializza il tensore di input per l'allenamento della rete neurale
  #
  # ogni campione contiene i valori di [PompaGlicoleMarcia, TemperaturaCelle, TemperaturaRitornoGlicole, TemperaturaMandataGlicole] in una
  # finestra temporale che si estende fino ai 60 minuti precedenti
  #
  # i campioni contengono inoltre i valori della temperatura di mandata del glicole in una finestra temporale che si estende fino ai 60
  # minuti successivi

  X = np.zeros(shape=(n_samples, window_size, n_columns-1))
  i = 0
  for window in ds.rolling(window=window_size):
    if i >= ds.shape[0] - window_size:
      break
    if i > window_size - 1:
      X[i-window_size:i-window_size+1] = window.iloc[:, :n_columns-1]
    i += 1
  X1 = np.zeros(shape=(n_samples, window_size, 1))
  i = 0
  for window in ds.rolling(window=window_size):
    if i > 2 * window_size - 1:
      X1[i-2*window_size:i-2*window_size+1] = window.iloc[:, n_columns-1:]
    i += 1

  # Tutti i valori vengono riscalati tra 0 ed 1
  scalers = {}
  for i in range(X.shape[1]):
    scalers[i] = MinMaxScaler()
    X[:, i, :] = scalers[i].fit_transform(X[:, i, :]) 

  scaler = MinMaxScaler(feature_range=(0, 1))
  Y = scaler.fit_transform(Y)

  return X, Y


# La funzione che definisce e compila il modello

def compile_model(input_shape):
  model = Sequential()
  model.add(LSTM(50, input_shape=(input_shape[0], input_shape[1])))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam')

  return model


ds = pd.read_csv(f"01 09 - 04 10/CSV_celle/{prefix}.csv")
X, Y = prepare_dataset(ds)

n_train = int(X.shape[0] * TRAIN_PERCENTAGE)
X_train, Y_train = X[:n_train], Y[:n_train]
X_val, Y_val = X[n_train:], Y[n_train:]

model = compile_model((X_train.shape[1], X_train.shape[2]))

history = model.fit(X_train, Y_train, epochs=10, batch_size=72, validation_data=(X_val, Y_val), verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()