import os
import re
import pandas as pd

# Script in Python che prende un csv e
# toglie informazioni superflue
# fa un cast corretto delle colonne (Le colonne che contengono solamente numeri interi vengono fatte diventare interi anzichè float)
# usa il punto al posto della virgola come separatore per i numeri decimali
# usa la virgola come separatore per i diversi valori anzichè il punto e virgola
# toglie la colonna "Unnamed"

# L'input viene preso dalla cartella "15min" oppure "1hr" e l'output viene messo nella cartella "better"

FILE_NAME = "all.csv"
FULL_INPUT_PATH = os.path.join("15min", FILE_NAME)
FULL_OUTPUT_PATH = os.path.join("better", FILE_NAME)

df = pd.read_csv(FULL_INPUT_PATH, sep=";", low_memory=False)

# Contiene una lista di regex, ogni regex matcha sempre e solo campi che contengono numeri interi
int_fields = [
    re.compile(r"Cell[ae].*Energia Attiva"),
    re.compile(r"Antigelo"),
    re.compile(r"Umidificazione"),
    re.compile(r"Aria"),
    re.compile(r"Ventilazione Forzata"),
]

# Drop unnamed column
df.drop(df.columns[df.columns.str.contains("Unnamed")], axis=1, inplace=True)


def make_better():
    output_data = {"Date": df["Date"]}

    # Per ogni colonna del dataset modifico la chiave in qualcosa più leggibile, separato
    # solamente da spazi anzichè da '_' e rimuovendo termini inutili come 'General' e 'Tecnico' 
    k: str
    for k in df:
        new_key = re.search(r"(Melinda ipogeo_(Lotto [0-9]+|General)_)(.*)", k)

        if new_key == None:
            continue
        else:
            new_key = new_key.group(3)

        # Rimuovi "General" (ma non "Generale") da qualsiasi stringa
        new_key = re.sub(r"General(?!e)", "", new_key)
        new_key = new_key.replace("Tecnico", "")
        new_key = new_key.replace("_", " ")
        # Sostituisci whitespace consecutivi con uno solo
        new_key = re.sub(r"\s+", " ", new_key).strip()

        new_values = []

        for i in range(len(df)):
            v = df[k][i]
            if pd.isnull(v):
                new_values.append(None)
            elif isinstance(v, str):
                new_values.append(v.replace(",", "."))
            elif any(regex.search(k) for regex in int_fields):
                # Se una qualsiasi delle regex matcha vuol dire che il campo deve restare intero
                new_values.append(int(v))
            else:
                # Altrimenti vuol dire che il campo deve essere convertito da int a float
                new_values.append(float(v))

        output_data[new_key] = new_values

    pd.DataFrame(output_data).to_csv(FULL_OUTPUT_PATH, index=False)


make_better()
