#!/usr/bin/python
import getopt
import os
import sys

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

import dataframe_importer as importer
from dataframe_importer import CellDataset
from RNNs import *
import matplotlib.pyplot as plt

import csv

argv = sys.argv[1:]
csv_file = ''

batch_size = 128
hidden_dim = 64
layer_dim = 3
dropout = 0.2
n_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-6
neural_network_kind = "lstm"
test = 0
plots = 1

try:
    opts, args = getopt.getopt(argv, "hnbHLelwdTpi:", ["network=", "input-file=", "batch=", "hidden-dim=", "layer-dim=",
                                                     "epochs=", "dropout=", "learning-rate=", "weight-decay=","test=",
                                                       "plots="])
except getopt.GetoptError:
    print("training(RNN).py -i csv_file_path")
    sys.exit(2)

for opt, arg in opts:
    if opt in ('-h', '--help'):
        print(f"""training(RNN).py -i csv_file_path
        Arguments: 
         -h, --help: elenca gli argomenti necessari
         -n, --network: uno tra rnn, lstm e gru (default: {neural_network_kind})
         -i, --input-file: percorso del file csv da passare come input
         -b, --batch: batch per il training (default: {batch_size})
         -H, --hidden-dim: numero di hidden layers (default: {hidden_dim})
         -L, --layer-dim: dimensione dei layer (default: {layer_dim})
         -e, --epochs: epoche per cui addestrare la rete (default: {n_epochs})
         -l, --learning-rate: rate di apprendimento della rete (default: {learning_rate})
         -w, --weight-decay: decadimento dei pesi (default: {weight_decay})
         -d, --dropout: dropout dei neuroni (default: {dropout})
        -t, --test: se si tratta di model assessment (0) o Model Selection (1). (default:{test})
        -p, --plots: se stampare i grafici (1) o meno (0). Default: {plots} """)
        sys.exit(0)
    elif opt in ("-i", "--ifile"):
        csv_file = arg
    elif opt in ("-n", "--network"):
        neural_network_kind = arg
    elif opt in ("-b", "--batch"):
        batch_size = int(arg)
    elif opt in ("-H", "--hidden-dim"):
        hidden_dim = int(arg)
    elif opt in ("-L", "--layer-dim"):
        layer_dim = int(arg)
    elif opt in ("-e", "--epochs"):
        n_epochs = int(arg)
    elif opt in ("-l", "--learning-rate"):
        learning_rate = float(arg)
    elif opt in ("-w", "--weight-decay"):
        weight_decay = float(arg)
    elif opt in ("-d", "--dropout"):
        dropout = float(arg)
    elif opt in ("-T", "--test"):
        test = int(arg)
    elif opt in ("-p", "--plots"):
        test = int(arg)

if csv_file == '':
    print("Missing argument: -i csv_path\nes: training(RNN).py -i csv_file_path")
    sys.exit(2)

converter = importer.CellCsvConverter(useful_variables=["PercentualeAperturaValvolaMiscelatrice",
                                                        "PercentualeVelocitaVentilatori",
                                                        "PompaGlicoleMarcia",
                                                        "Raffreddamento",
                                                        "TemperaturaCelle",
                                                        "TemperaturaMandataGlicole",
                                                        "TemperaturaRitornoGlicole",
                                                        "UmiditaRelativa",
                                                        "VentilatoreMarcia"])
train_dataset, validation_dataset, test_dataset = converter.Convert_csv_to_Dataset(csv_file, train_percentage=0.7,
                                                                                   normalize=False)


# check again for nan
def remove_nan(dataset: CellDataset):
    nan_id = list(set(dataset.x_train.isnan().argwhere().numpy()[:, 0]))
    dataset.x_train = dataset.x_train[~torch.any(dataset.x_train.isnan(), dim=1)]

    if nan_id:
        a = dataset.y_train.numpy()
        a = np.delete(a, nan_id)
        dataset.y_train = torch.tensor(a)

    return dataset


train_dataset = remove_nan(train_dataset)
validation_dataset = remove_nan(validation_dataset)
test_dataset = remove_nan(test_dataset)

# data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

input_dim = train_dataset.x_columns
val_dim = validation_dataset.x_columns
output_dim = 1

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'dropout_prob': dropout}

model = get_model(neural_network_kind, model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader=train_loader, val_loader=val_loader, batch_size=batch_size, n_epochs=n_epochs,
          train_features=input_dim,
          val_features=val_dim)
if plots == 1:
    opt.plot_losses()

model_loss = opt.last_validation_loss()

data = {
    'final_validation_loss':model_loss,
    'network':neural_network_kind,
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'layer_dim': layer_dim,
    'output_dim': output_dim,
    'dropout_prob': dropout,
    'learning_rate':learning_rate,
    'epochs':n_epochs,
    'weight_decay':weight_decay
}

cartella = "./cross_validation_results/cella_"+csv_file[-6:-4]+"/"
file = cartella + neural_network_kind + '.csv'
if not os.path.exists("./cross_validation_results/cella_"+csv_file[-6:-4]+"/"):
    os.mkdir(cartella)
with open(file, mode="a") as f:
    writer = csv.DictWriter(f, data.keys())
    if os.stat(file).st_size == 0:
        writer.writeheader()
    writer.writerow(data)
# TODO: continue

predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)


def format_predictions(predictions, values):
    return pd.DataFrame(data={"value": values, "prediction": predictions})


def calculate_metrics(df):
    return {'mae': mean_absolute_error(df.value, df.prediction),
            'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2': r2_score(df.value, df.prediction)}


# TODO: rendere carino e fixare
def plot_data(df:pd.DataFrame, metrics):
    # x = range(0, len(df)) # Sample data.
    plt.plot(df.prediction, label="Predizione")
    plt.plot(df.value, label="Valori reali")
    plt.legend()
    plt.title("Grafico previsioni")
    # plt.text(0, 0, f"MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, R2: {metrics['r2']}")
    plt.show()
    plt.close()


df = format_predictions(predictions, values)
result_metrics = calculate_metrics(df)
if plots == 1:
    plot_data(df, result_metrics)

results = {
    "network": neural_network_kind,
    "Mean Absolute Error": result_metrics['mae'],
    'Root Mean Square Error': result_metrics['rmse'],
    'R2': result_metrics['r2']
}

print(results)
if test == 1:
    file = "./test_modelli.csv"
    with open(file, mode="a") as f:
        writer = csv.DictWriter(f, data.keys())
        if os.stat(file).st_size == 0:
            writer.writeheader()
        writer.writerow(data)
