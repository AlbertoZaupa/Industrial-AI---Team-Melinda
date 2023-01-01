""" 
This module contains the function to import data from the specified cell number. 

@author: Matteo Dalle Vedove (matteodv99tn@gmail.com)
"""

import pandas as pd
from constants import ORIGINAL_DATA_PATH, PROCESSED_DATA_PATH
from os.path import join, isfile

def import_data(cell_number: int, processed_data: bool = True) -> pd.DataFrame:
    """ Imports data from the specified cell number.

    Args:
        cell_number (int): The cell number to import data from.
        processed_data (bool, optional): Whether to import processed data or not. Defaults to True.

    Returns:
        pd.DataFrame: The imported data.
    """

    if processed_data:
        base_path = PROCESSED_DATA_PATH
    else:
        base_path = ORIGINAL_DATA_PATH
    file_path = join(base_path, f'Cella_{cell_number}.csv')

    if(not isfile(file_path)):
        raise FileNotFoundError(f'File "Cella_{cell_number}.csv" do not exists in "{base_path}"')

    print('Loading', file_path)

    df = pd.read_csv(file_path, parse_dates = ['Date'])
    df = df.dropna()
    df = df.reset_index(drop = True)

    return df


if __name__ == '__main__':
    df = import_data(13, False)