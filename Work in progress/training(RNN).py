#!/usr/bin/python
import getopt
import sys
import torch.optim as optim
from torch.utils.data import DataLoader

import dataframe_importer as importer
from RNNs import *

argv = sys.argv[1:]
csv_file = ''
try:
    opts, args = getopt.getopt(argv, "hi", ["ifile="])
except getopt.GetoptError:
    print("training(RNN).py -i csv_file_path")
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('training(RNN).py -i csv_file_path')
        sys.exit(0)
    elif opt in ("-i", "--ifile"):
        csv_file = arg

converter = importer.CellCsvConverter()
train_dataset, test_dataset = converter.Convert_csv_to_Dataset(csv_file, train_percentage=0.7, validation=True)

# data loaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

input_dim = len(X_train.columns)
output_dim = 1
hidden_dim = 64
layer_dim = 3
batch_size = 64
dropout = 0.2
n_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'dropout_prob': dropout}

model = get_model('lstm', model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
