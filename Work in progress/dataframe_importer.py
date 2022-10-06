"""
Usage: script per la classe converter
"""
import pandas as pd
import torch
from torch.utils.data import Dataset


# this assumes that we already have the single cell CSV
class CellDataset(Dataset):
    """Classe per Pytorch datasets"""

    def __init__(self, dataframe: pd.DataFrame, y_name="TemperaturaCelle"):
        train_x = dataframe.drop(y_name, axis=1).values # seleziona tutti i valori tranne quelli della variabile che vogliamo fittare
        train_y = dataframe[y_name].values # seleziona solo i valori della variabile che vogliamo fittare

        # trasforma in tensori di PyTorch
        self.x_train = torch.tensor(train_x)
        self.y_train = torch.tensor(train_y)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class CellCsvConverter:
    """Lo scopo di questa classe è convertire il csv di una cella in un dataframe pandas,
    in modo che sia pronto per l'analisi"""

    def __init__(self, useful_variables=None, y="TemperaturaCelle", delimiter=","):
        """Inizializza il convertitiore.
                @:param y: variabile da fittare
                @:param delimiter: delimitatore all'interno dei csv. Dovrebbero essere sempre virgole, ma per sicurezza.
                @:param useful_variables: permette di indicare le variabili utili, ovvero quelle con cui creare il dataframe.
                        Se si imposta useful_variables = 'All', verranno prese in considerazione tutte le variabili.
                        Lista completa delle variabili: https://github.com/AlbertoZaupa/Industrial-AI---Team-Melinda/blob/master/Note/dataset.md """
        self.y = y
        self.delimiter = delimiter
        if useful_variables is None:
            useful_variables = ['TemperaturaCelle', 'TemperaturaMandataGlicole',
                                'TemperaturaRitornoGlicole', 'TemperaturaMele',
                                'PercentualeAperturaValvolaMiscelatrice',
                                'TemperaturaRoccia1', 'TemperaturaRoccia2', 'TemperaturaRoccia3',
                                'TemperaturaRoccia4', 'Preventilazione', 'Postventilazione',
                                'PompaGlicoleMarcia',
                                'PercentualeVelocitaVentilatori']

        self.useful_variables = useful_variables

    def Convert_csv_to_Dataframe(self, csv_path: str) -> pd.DataFrame:
        """Converte il csv in Dataframe di Pandas"""
        dataset = pd.read_csv(csv_path, delimiter=self.delimiter)
        if self.useful_variables == "All":
            return dataset
        else:
            return dataset[self.useful_variables]

    def Convert_csv_to_Dataset(self, csv_path: str, train_percentage=0.8) -> (Dataset, Dataset):
        """Converte il csv direttamente in Train e Test dataset. Se la train percentage è 1 allora ritorna la tupla (Train dataset, None)
           NB: i dati vengono importati senza shuffle"""
        dataframe = self.Convert_csv_to_Dataframe(csv_path) # converti il csv in pandas dataframe

        train_number = int(len(dataframe.index) * train_percentage) # serve a splittare. Essendo fatto per le RNN, non viene effettuato lo shuffle
        train_dataset = CellDataset(dataframe.iloc[0:train_number], self.y)
        if train_dataset == 1.0:
            return train_dataset, None
        test_dataset = CellDataset(dataframe.iloc[train_number:], self.y)

        return train_dataset, test_dataset


# test
# __name__ = "main"  # s/commentare questo
if __name__ == "main":
    converter = CellCsvConverter()
    df = converter.Convert_csv_to_Dataframe("./data/Cella_15.csv")
    print(df)

    dstrain, dstest = converter.Convert_csv_to_Dataset("./data/Cella_15.csv")
    print(dstrain)
    print(dstest)
