import os
import pandas as pd
import numpy as np

# Questo script prende in input ./better/all.csv e genera un file ./better/all_nn.csv
# all_nn.csv è un file più piccolo che contiene solamente le colonne che interessano al neural network
# e che quindi è più veloce da caricare e gestire

INPUT_FULL_PATH = os.path.join("better", "all.csv")
OUTPUT_FULL_PATH = os.path.join("better", "all_nn.csv")

df = pd.read_csv(INPUT_FULL_PATH)

columns = [
    "Apertura Valvola Miscelatrice",
    "Marcia Pompa Glicole",
    "Marcia Ventilatore",
    "Temperatura Cella",
    "Temperatura Mandata Glicole",
    "Temperatura Mele",
    "Temperatura Ritorno Glicole",
    "Temperatura Roccia 1",
    "Temperatura Roccia 2",
    "Temperatura Roccia 3",
    "Temperatura Roccia 4",
]

output_df = pd.DataFrame()
output_df["Date"] = df["Date"]

for numero_cella in range(1, 35):
    for col in columns:
        k = f"Cella {numero_cella} {col}"
        output_df = pd.concat([output_df, df[k]], axis=1)

keys_valvole = [f"Cella {numero_cella} {columns[0]}" for numero_cella in range(1, 35)]
keys_marcia_pompa = [
    f"Cella {numero_cella} {columns[1]}" for numero_cella in range(1, 35)
]
sum_valvole = []
valvole_aperte = []

for row_idx in range(len(output_df)):
    valori_valvole = output_df[keys_valvole].iloc[row_idx]
    sum_valvole.append(np.sum(valori_valvole))
    valori_marcia_pompa = output_df[keys_marcia_pompa].iloc[row_idx]
    valvole_aperte.append(int(np.sum(valori_marcia_pompa)))

output_df.insert(1, "Somma Apertura Valvola Miscelatrice", sum_valvole)
output_df.insert(2, "Numero Valvole Aperte", valvole_aperte)

output_df.to_csv(OUTPUT_FULL_PATH, index=False)
