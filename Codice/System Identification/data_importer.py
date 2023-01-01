""" 
This module contains the function to import data from the specified cell number. 

@author: Matteo Dalle Vedove (matteodv99tn@gmail.com)
"""

import pandas as pd
import matplotlib.pyplot as plt
from constants import ORIGINAL_DATA_PATH, PROCESSED_DATA_PATH
from os.path import join, isfile
from datetime import datetime

def import_data(cell_number: int, processed_data: bool = True) -> pd.DataFrame:
    """ Imports data from the specified cell number. Furthermore, it adds a new column "Minuti" 
    which contains the time (in minutes) from the first measurement.

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
    df['Minuti'] = (df['Date'] - df.Date[0]).apply(lambda x: x.total_seconds() / 60)

    return df

def show_cell_data( df: pd.DataFrame, 
                    begin: datetime = None, 
                    end: datetime = None, 
                    size: tuple = (15, 10)):
    """ Creates a plot of the data of the specified cell.

    Args:
        df (pd.DataFrame): dataframe containing the data to show.
        begin (datetime, optional): the start date. If not set, it shows starting from the first 
            entry in the dataframe.
        end (datetime, optional): the end date. If not set, is ends 7 days after the beginning.
        size (tuple, optional): the size of the plot. Defaults to (15, 10).
    """

    if begin is None:
        begin = df.Date[0]
    if end is None:
        end = begin + pd.Timedelta(days = 7)

    df = df[(df.Date >= begin) & (df.Date <= end)]

    fig, ax = plt.subplots(2, 1, figsize = (15, 10), sharex=True)

    ax[0].plot(df.Date, df.TemperaturaMandataGlicole, label = 'Mandata Glicole')
    ax[0].plot(df.Date, df.TemperaturaRitornoGlicole, label = 'Ritorno Glicole')
    ax[0].plot(df.Date, df.TemperaturaCelle,          label = 'Celle')
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Temperatures [°C]')
    ax[0].legend(loc='upper right')
    ax[0].grid(which='major', linestyle='dashed', linewidth='0.5', color='lightgrey')

    ax[1].plot(df.Date, 100*df.PompaGlicoleMarcia,                 label = 'Pompa glicole (on/off)')
    ax[1].plot(df.Date, df.PercentualeAperturaValvolaMiscelatrice, label = '% valvola miscelatrice')
    ax[1].grid(which='major', linestyle='dashed', linewidth='0.5', color='lightgrey')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel('Date')
    

    return fig

if __name__ == '__main__':
    df = import_data(13, False)
    show_cell_data(df)
    plt.show()