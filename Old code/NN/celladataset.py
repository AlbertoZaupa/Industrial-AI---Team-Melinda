import pandas as pd
import torch
from torch.utils.data import Dataset

# Definisce il dataset pytorch che consente di prendere i dati da "Valori/better/all_nn.csv"
# Un dataset deve fornire al minimo 3 metodi per poter funzionare: __init__, __len__ e __get_item__

class CellaDataset(Dataset):
    # Definisce quanti quarti d'ora guardare indietro per un sample
    # Se lookbehind == 4 allora vuol dire che guardo i dati di 1 ora per determinare
    # l'apertura della valvola nel quarto d'ora successivo
    look_behind = 4

    def __init__(self, file_path, numero_celle, columns, months = None, filter=None):
        """
        file_path: str (il file path del file csv, "Valori/better/all_nn.csv")
        numero_celle: List[int] (il numero delle celle che voglio che vengano usate nel dataset)
        columns: List[str] (le colonne che voglio vengano usate nel dataset, la colonna con indice 0 è la colonna che voglio usare come label, quindi Apertura Valvola Miscelatrice)
        months: List[int] (il numero dei mesi che voglio che vengano usate nel dataset)
        filter: (None, "zero", "nonzero") se filter è "zero" allora vengono presi sample solamente di dati che hanno Apertura Valvola Miscelatrice == 0, se filter è "nonzero" allora il dataset fornisce dei sample che hanno sempre Apertura Valvola Miscelatrice != 0
        """
        self.numero_celle = numero_celle.copy()
        self.columns = columns.copy()
        self.filter = filter
        # Nella classe viene tenuto l'intero dataset, e viene memorizzata una lista di interi che contiene
        # gli indici delle righe del dataset che corrispondono ai criteri che ho specificato nel costruttore
        # (quindi conterrà solamente le righe dei numeri delle celle giuste, dei mesi giusti, e dei filtri "zero" o "nonzero" giusti)
        self.indexes = []
        # self.df conterrà le colonne dell'intero dataset che ho specificato
        # A self.df vengono aggiunte delle righe di "Data" e "Numero Cella" sempre e comunque
        # anche se non vengono specificate in "columns"
        self.df = pd.DataFrame()
        file_df = pd.read_csv(file_path)
        
        # Filtro indexes e tengo solo quelli che hanno alla riga associata il mese corrispondente
        if months is not None:
            # Parsing del campo Date in datetime
            file_df["Date"] = pd.to_datetime(file_df["Date"], format="%d-%m-%Y %H:%M:%S")
            # Filtro solamente le righe che hanno valori del mese in months
            file_df = file_df[file_df["Date"].dt.month.isin(months)]
            file_df.reset_index(drop=True, inplace=True)

        # Aggiungo i valori di ogni cella a self.df
        for numero_cella in self.numero_celle:
            cella_df = pd.DataFrame()
            cella_df["Date"] = file_df["Date"]
            cella_df["Numero Cella"] = numero_cella
            for k in self.columns:
                cella_df[k] = file_df[f"Cella {numero_cella} {k}"]

            wrong_indexes = set()
            for idx in [-1] + cella_df[cella_df.isnull().any(axis=1)].index.tolist():
                for x in range(CellaDataset.look_behind + 1):
                    wrong_indexes.add(idx + x)
            wrong_indexes.remove(-1)
            # To check all the rows that are not in wrong_indexes use
            # df[~df.index.isin(wrong_indexes)]

            for idx in range(len(cella_df)):
                if idx not in wrong_indexes:
                    self.indexes.append(len(self.df) + idx)
            self.df = pd.concat([self.df, cella_df], ignore_index=True)

        self.df.reset_index(drop=True, inplace=True)

        # Filtro indexes e tengo solamente quelli che hanno nella riga index un valore == 0 o != 0
        if filter == "zero":
            temp = self.df.iloc[self.indexes]
            self.indexes = temp[temp[self.columns[0]] == 0].index.tolist()
        elif filter == "nonzero":
            temp = self.df.iloc[self.indexes]
            self.indexes = temp[temp[self.columns[0]] != 0].index.tolist()


    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        '''
        Prende un indice nel range [0, len(self.indexes)] e restituisce un sample
        il sample che viene ritornato è un sample che corrisponde ai criteri
        che sono stati specificati nel costruttore del dataset
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.__get_sample(self.indexes[idx])

        return sample

    def __get_sample(self, idx):
        '''
        Prende un indice nel range [look_behind, len(self.df)]
        e permette di ottenere un sample che comprende i look_behind quarti
        d'ora precedenti e la label del quarto d'ora successivo 
        '''
        end_idx = idx
        start_idx = idx - CellaDataset.look_behind

        # Estraggo le righe in range [start_idx, end_idx]
        rows = self.df.iloc[start_idx : end_idx + 1][self.columns]

        # Quello che voglio prevedere è l'apertura della valvola
        # Prelevo il parametro e lo salvo in label
        label = torch.tensor(rows.iloc[-1][self.columns[0]])
        label = label.float()
        label = label.unsqueeze(0)
        # rows.drop(self.columns[0], axis=1, inplace=True)
        rows.drop(index=rows.index[-1], axis=0, inplace=True)

        # L'input è "rows", converto tutto in un tensore
        input = torch.tensor(rows.to_numpy())
        input = input.float()
        input = input.unsqueeze(0)
        if torch.any(input.isnan()):
            print(f"Input alla riga {idx} ha un valore NaN in qualche posizione")

        sample = {"input": input, "label": label}

        return sample

    def to_csv(self, file_path, model, device="cpu", round_prediction=False):
        '''
        Genera il file csv con i dati del dataset passati in input e le predizioni che vengono fatte dal modello passato in input
        round_prediction consente di fare l'arrotondamento del valore che viene generato dal modello
        (nel caso si volesse prevedere per esempio "Marcia Pompa Glicole" invece di "Apertura Valvola Miscelatrice")
        '''
        with torch.no_grad():
            output_df = pd.DataFrame()

            output_df["Date"] = self.df["Date"]
            output_df["Numero Cella"] = self.df["Numero Cella"]
            for col in self.columns[1:]:
                output_df[col] = self.df[col]
            output_df[self.columns[0]] = self.df[self.columns[0]]
            predictions = [None] * len(self.df)

            for idx in self.indexes:
                input: torch.Tensor = self.__get_sample(idx)["input"].to(device)
                output: torch.Tensor = model(input)
                predictions[idx] = output.item()

            if round_prediction:
                for i in range(len(predictions)):
                    if predictions[i] is None:
                        continue
                    predictions[i] = round(predictions[i])

            output_df["Prediction"] = predictions
            output_df.to_csv(file_path, index=False)
