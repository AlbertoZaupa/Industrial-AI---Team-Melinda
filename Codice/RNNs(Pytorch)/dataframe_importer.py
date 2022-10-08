"""
Usage: script per la classe converter
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
pd.options.mode.chained_assignment = None  # default='warn'


# this assumes that we already have the single cell CSV
class CellDataset(Dataset):
    """Classe per Pytorch datasets"""

    def __init__(self, dataframe: pd.DataFrame, y_name="TemperaturaCelle"):
        train_x = dataframe.drop(y_name,
                                 axis=1).values  # seleziona tutti i valori tranne quelli della variabile che vogliamo fittare
        train_y = dataframe[y_name].values  # seleziona solo i valori della variabile che vogliamo fittare

        # trasforma in tensori di PyTorch
        self.x_train = torch.tensor(train_x)
        self.y_train = torch.tensor(train_y)
        self.rows = len(self.y_train)
        self.x_columns = len(dataframe.columns) - 1

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
            useful_variables = ["PercentualeAperturaValvolaMiscelatrice"
                ,"PercentualeVelocitaVentilatori",
                                "PompaGlicoleMarcia",
                                "Raffreddamento",
                                "TemperaturaCelle",
                                "TemperaturaMandataGlicole",
                                "TemperaturaRitornoGlicole",
                                "UmiditaRelativa",
                                "VentilatoreMarcia"]

        self.useful_variables = useful_variables

    def Convert_csv_to_Dataframe(self, csv_path: str) -> pd.DataFrame:
        """Converte il csv in Dataframe di Pandas"""
        dataset = pd.read_csv(csv_path, delimiter=self.delimiter)
        if self.useful_variables == "All":
            return dataset
        else:
            return dataset[self.useful_variables]

    def Sanitize(self, dataframe:pd.DataFrame, normalize:bool):

        def Deal_with_na(dataframe:pd.DataFrame):
            return dataframe.dropna()

        def normalize_helper(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        def Normalize(dataframe:pd.DataFrame):

            for (column_name, column_data) in dataframe.iteritems():
                if column_name == self.y:
                    continue
                dataframe.loc[:, column_name] = normalize_helper(column_data)
                if dataframe[column_name].isnull().any(): # erano tutti valori uguali
                    dataframe.loc[:,column_name] = dataframe[column_name].fillna(0)

            return dataframe

        dataframe = Deal_with_na(dataframe)
        if normalize:
            dataframe = Normalize(dataframe)
        return dataframe

    def Convert_csv_to_Dataset(self,
                               csv_path: str,
                               train_percentage=0.8,
                               validation_percentage=0.2,
                               normalize=True) -> (Dataset, Dataset, Dataset):
        """Converte il csv direttamente in Train, Validation e Test dataset.
           NB: i dati vengono importati senza shuffle"""
        dataframe = self.Convert_csv_to_Dataframe(csv_path)  # converti il csv in pandas dataframe
        if normalize:
            self.Sanitize(dataframe,True)
        validation_dataset = None
        test_dataset = None

        train_number = int(
            len(dataframe.index) * train_percentage)  # serve a splittare. Essendo fatto per le RNN, non viene effettuato lo shuffle
        train_dataset = CellDataset(dataframe.iloc[0:train_number], self.y)
        train_rows = len(train_dataset)
        if validation_percentage != 0:
            train_percentage = 1 - validation_percentage
            train_number = int(train_rows * train_percentage)
            train_dataset = CellDataset(dataframe.iloc[0:train_number], self.y)
            validation_dataset = CellDataset(dataframe.iloc[train_number:train_rows], self.y)
        if train_dataset != 1.0:
            test_dataset = CellDataset(dataframe.iloc[train_rows:], self.y)

        return train_dataset, validation_dataset, test_dataset


# test
# __name__ = "main"  # s/commentare questo
if __name__ == "main":
    converter = CellCsvConverter()
    df = converter.Convert_csv_to_Dataframe("./data/Cella_15.csv")
    print(df)

    dstrain, dsval, dstest = converter.Convert_csv_to_Dataset("./data/Cella_15.csv")
    print(dstrain)
    print(dstest)
