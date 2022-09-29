import os
import pandas as pd

# Script che prende un file csv con i dati dell'ipogeo di Melinda e
# genera le due colonne di temperatura ritorno glicole e
# apertura valvola miscelatrice

FULL_PATH = os.path.join(".", "better", "all.csv")

df = pd.read_csv(FULL_PATH)

columns = ["Cella {} Temperatura Ritorno Glicole",
           "Cella {} Apertura Valvola Miscelatrice"]

output = pd.DataFrame()

for cella in range(1, 35):
    for col in columns:
        key = col.format(cella)
        output[key] = df[key]

output.to_csv("two_columns.csv") 
