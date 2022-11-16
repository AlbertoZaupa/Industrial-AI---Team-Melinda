import pandas as pd

DATASET_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets/Cella_13.csv"
SH_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets/Cella_13_SH.csv"
OUTPUT_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets/Cella_13_SH_merged.csv"

if __name__ == "__main__":
    cell13_df = pd.read_csv(DATASET_PATH)
    sh_values = pd.read_csv(SH_PATH)[["0", "1"]]
    sh_values = sh_values.values
    cell13_df["Setpoint"] = sh_values[:, :1]
    cell13_df["Isteresi"] = sh_values[:, 1:]
    cell13_df.to_csv(OUTPUT_PATH)

