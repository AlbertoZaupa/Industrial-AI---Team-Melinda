import os
import pandas as pd

# Script in python che prende in input i dataset csv di diversi mesi
# e genera un file unico concatenando i diversi mesi nel file di output

months = [
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
]

OUTPUT_FILE_FULL_PATH = os.path.join("15min", "all.csv")
INPUT_FILES_FULL_PATH = [
    os.path.join("15min", f"{month}_15min.csv") for month in months
]

output_df = pd.DataFrame()

for file_path in INPUT_FILES_FULL_PATH:
    df = pd.read_csv(file_path, sep=";", low_memory=False)
    output_df = pd.concat([output_df, df])

output_df.to_csv(OUTPUT_FILE_FULL_PATH, sep=";", index=False)
