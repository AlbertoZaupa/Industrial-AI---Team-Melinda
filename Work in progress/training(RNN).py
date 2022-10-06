#!/usr/bin/python
import getopt
import sys
import torch.optim as optim
from torch.utils.data import DataLoader

import dataframe_importer as importer
from RNNs import *

argv = sys.argv[1:]
csv_file = ''

batch_size = 128
hidden_dim = 64
layer_dim = 3
dropout = 0.2
n_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-6
neural_network_kind = "lstm"

try:
    opts, args = getopt.getopt(argv, "hni:bHLelwd", ["network=", "input-file=", "batch=", "hidden-dim=", "layer-dim=",
                                                     "epochs=", "dropout=", "learning-rate=", "weight-decay"])
except getopt.GetoptError:
    print("training(RNN).py -i csv_file_path")
    sys.exit(2)

for opt, arg in opts:
    if opt in ('-h', '--help'):
        print(f"""training(RNN).py -i csv_file_path
        Arguments: 
         -h, --help: elenca gli argomenti necessari
         -n, --neural-network: uno tra rnn, lstm e gru (default: {neural_network_kind})
         -i, --input-file: percorso del file csv da passare come input
         -b, --batch: batch per il training (default: {batch_size})
         -H, --hidden-dim: numero di hidden layers (default: {hidden_dim})
         -L, --layer-dim: dimensione dei layer (default: {layer_dim})
         -e, --epochs: epoche per cui addestrare la rete (default: {n_epochs})
         -l, --learning-rate: rate di apprendimento della rete (default: {learning_rate})
         -w, --weight-decay: decadimento dei pesi (default: {weight_decay})
         -d, --dropout: dropout dei neuroni (default: {dropout})""")
        sys.exit(0)
    elif opt in ("-i", "--ifile"):
        csv_file = arg
    elif opt in ("-n", "--neural-network"):
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

converter = importer.CellCsvConverter()
train_dataset, validation_dataset, test_dataset = converter.Convert_csv_to_Dataset(csv_file,
                                                                                   train_percentage=0.7,
                                                                                   validation_percentage=0.2)


# data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

input_dim = len(train_dataset)
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
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
