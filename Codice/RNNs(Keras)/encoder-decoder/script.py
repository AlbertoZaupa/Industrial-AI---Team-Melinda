import pandas as pd
import matplotlib.pyplot as plt
from dataset import prepare_dataset
from model import encoder_decoder

# Le colonne che vengono utilizzate durante il training
# TODO sostituire 'Temperatura mandata glicole' con 'Temperatura glicole nominale'
columns = [
    # LA PRIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI VIENE PREDETTO IL VALORE
    "TemperaturaRitornoGlicole",
    "TemperaturaCelle",
    "PompaGlicoleMarcia",
    "PercentualeAperturaValvolaMiscelatrice",
    # L'ULTIMA COLONNA IN QUESTA LISTA E' LA VARIABILE DI CUI IL MODELLO RICEVE I VALORI FUTURI IN INPUT
    "TemperaturaMandataGlicole",
]
prefix = "Ipogeo2_Cella13"
columns = [prefix + column for column in columns]
n_columns = len(columns)
target_column = columns[0]


df = pd.read_csv(f"01 09 - 04 10/CSV_celle/{prefix}.csv")
df = df[columns]
df = df.replace({',': '.'}, regex=True).astype(float)
df = df.fillna(0) # TODO migliorare la modalità tramite cui eliminiamo i N/A (ad esempio sostituendoli con una media locale)


# Scaling nel range [0, 1]
scaling_factors = {column: (df[column].min(), df[column].max()) for column in df.columns}
scaled_df = pd.DataFrame()
for column in df.columns:
  scaled_df[column] = (df[column] - scaling_factors[column][0])/(scaling_factors[column][1] - scaling_factors[column][0])


# training, validation e test split
df_len = df.shape[0]
TRAIN_SPLIT = int(0.7 * df_len)
VALIDATION_SPLIT = int(0.2 * df_len)


train_df = scaled_df.iloc[:TRAIN_SPLIT]
val_df = scaled_df.iloc[TRAIN_SPLIT:TRAIN_SPLIT + VALIDATION_SPLIT]
test_df = scaled_df.iloc[TRAIN_SPLIT + VALIDATION_SPLIT:]


# viene plottato il valore della variabile target
x_axis = range(test_df.shape[0])
plt.plot(x_axis, test_df[target_column]*scaling_factors[target_column][1])
plt.show()


# vengono preparati i dataset
N_INPUT_FEATURES = 1
PAST_WINDOW_SIZE = 12*60
FORECAST_SIZE = 1*60
BATCH_SIZE = 512
train_data = prepare_dataset(train_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
validation_data = prepare_dataset(val_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)
test_data = prepare_dataset(test_df, N_INPUT_FEATURES, PAST_WINDOW_SIZE, FORECAST_SIZE, BATCH_SIZE)


# la rete neurale
model = encoder_decoder(PAST_WINDOW_SIZE, len(columns), FORECAST_SIZE, N_INPUT_FEATURES)


# la rete neurale viene allenata sul dataset di training
history = model.fit(train_data, epochs=25, validation_data=validation_data, verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# viene plottato il confronto tra i valori predetti dalla rete neurale ed i valori reali
scaling_factor = scaling_factors[target_column][1]

truth_flat = []
pred_flat = []
for batch in test_data:
  (past, future), truth = batch
  truth = truth * scaling_factor
  pred = model.predict((past,future)) * scaling_factor
  for v in truth[:, -1]:
    truth_flat.append(v)
  for v in pred[:, -1]:
    pred_flat.append(v)

plt.plot(truth_flat, label="Truth")
plt.plot(pred_flat, label="Prediction")
plt.legend()
plt.show()