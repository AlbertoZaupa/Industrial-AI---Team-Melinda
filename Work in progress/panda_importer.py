"""
Usage: script per la classe converter
"""
import pandas as pd


# this assumes that we already have the single cell CSV

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
        self.delimiter = ","
        if useful_variables is None:
            useful_variables = ['TemperaturaCelle', 'TemperaturaMandataGlicole',
                                'TemperaturaRitornoGlicole', 'TemperaturaMele',
                                'PercentualeAperturaValvolaMiscelatrice',
                                'TemperaturaRoccia1', 'TemperaturaRoccia2', 'TemperaturaRoccia3',
                                'TemperaturaRoccia4', 'Preventilazione', 'Postventilazione',
                                'PompaGlicoleMarcia',
                                'PercentualeVelocitaVentilatori']

        self.useful_variables = useful_variables

    def Convert(self, csv_path: str) -> pd.DataFrame:
        dataset = pd.read_csv(csv_path, delimiter=self.delimiter)
        if self.useful_variables == "All":
            return dataset
        else:
            return dataset[self.useful_variables]


# test
__name__ = "main"
if __name__ == "main":
    converter = CellCsvConverter()
    df = converter.Convert("./data/Cella_15.csv")
    print(df)