import pandas as pd
from constants import PURE_DATA_DIR, DATA_DIR
import os

csv_files_list = os.listdir(PURE_DATA_DIR)

def process_csv_file(filename: str) -> None:
  """
  Extracts relevant features from the all data of the csv file
  """
  pass
  # TO BE DONE:
  # Open the file (in PURE_DATA_DIR) and select useful features and save the processed result into a file into a proper folder (DATA_DIR)
  


# Perform the process_csv_file function on each original csv file
for file_name in csv_files_list:
  print(f"Processing file '{file_name}'")
  process_csv_file(os.path.join(PURE_DATA_DIR, file_name))


