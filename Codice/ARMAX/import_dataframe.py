""" 
Funzioni di base per leggere i dati delle celle e fare plot di base di confronto
o informativi.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os

## ---- SETUP ----
FILE_NAME = 'Cella_13.csv'
ORIGINAL_DATA = True


## ---- CODE ----
MODIFIER    = 'old' # subfoled inside of CSV
FILE_DIR    = str(os.path.realpath(os.path.dirname(__file__)))
tmp         = FILE_DIR[:FILE_DIR.rfind(os.sep)]
ROOT_DIR    = tmp[:tmp.rfind(os.sep)]
CSV_DIR     = os.path.join(ROOT_DIR, 'CSV')


def read_cell(filename: str, load_original: bool = True) -> pd.DataFrame:
    """
    In questo caso il flag "load_original" serve a specificare se caricare i 
    dati originali o quelli pre-processati da "cell_preprocessing.py".
    """
    from datetime import datetime

    if load_original:
        modifier = MODIFIER
        path = os.path.join(CSV_DIR, modifier, filename)
        print('Loading file', path)
        dateparse = lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
        df = pd.read_csv(
                path,
                parse_dates=['Date'], date_parser=dateparse
            )
        df = df.sort_values("Date")
        df = df.dropna()
        df = df.reset_index(drop=True)
    else:
        modifier = 'modified'
        path = os.path.join(CSV_DIR, modifier, filename)
        print('Loading file', path)
        df = pd.read_csv(
                path,
                parse_dates=['Date']
            )
    df.dropna()
    return df

def plot_cell(df: pd.DataFrame, fig_size=(9,7)):
    
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=fig_size )

    if 'Minuti' in df:
        time = 'Minuti'
    else:
        time = 'Date'

    ax[0].plot(df[time], df['TemperaturaMandataGlicole'], label='Mandata')
    ax[0].plot(df[time], df['TemperaturaRitornoGlicole'], label='Ritorno')
    ax[0].plot(df[time], df['TemperaturaMandataGlicoleNominale'], label='Nominale')
    ax[0].plot(df[time], df['TemperaturaCelle'], label='Cella')
    ax[0].legend()
    ax[0].set_title('Cella - sensori temperatura')
    ax[0].grid()

    ax[1].plot(df[time], 100*df['PompaGlicoleMarcia'], label='Pompa', linestyle='dashed')
    ax[1].plot(df[time], df['PercentualeAperturaValvolaMiscelatrice'], label='% Valvola')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title('Cella - raffreddamento')

    ax[2].plot(df[time], df['SalaMacchineTempCentraleSerbatorioGlicole'], label='Centrale')
    ax[2].plot(df[time], df['SalaMacchineTempManSerbatorioGlicole'], label='Mandata')
    ax[2].plot(df[time], df['SalaMacchineTempRitSerbatorioGlicole'], label='Ritorno')
    ax[2].legend()
    ax[2].grid()
    ax[2].set_title('Serbatoio - temperature')


def plot_correlation_matrix(df: pd.DataFrame, fig_size=(5,5)) -> None:
    fig, ax = plt.subplots(figsize=fig_size)
    corr_matrix = df.corr()
    sb.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    ind1 = df.columns.get_loc('TemperaturaMandataGlicoleNominale')
    ind2 = df.columns.get_loc('TemperaturaCelle')
    print('Correlazione TemperaturaMandataGlicoleNominale-TemperaturaCelle: ', df.corr().iloc[ind1, ind2])

def plot_comparison_cell(filename: str, fig_size=(10,10), dottedplot=True) -> None:
    from math import ceil, floor

    df_orig = read_cell(filename, True)
    df_mod = read_cell(filename, False)
    
    compare_labels = [
        'TemperaturaMandataGlicole',
        'TemperaturaMandataGlicoleNominale',
        'TemperaturaRitornoGlicole',
        'TemperaturaCelle',
        'PercentualeAperturaValvolaMiscelatrice',
        'SalaMacchineTempCentraleSerbatorioGlicole',
        'SalaMacchineTempManSerbatorioGlicole',
        'SalaMacchineTempRitSerbatorioGlicole'
        ]
    
    if dottedplot:
        mrk = '.'
    else:
        mrk = '-'
    fig, ax = plt.subplots(ceil(len(compare_labels)/2), 2, 
            sharex=True, figsize=fig_size
        )
    
    for idx, lab in enumerate(compare_labels):
        cax = ax[floor(idx/2), idx%2]
        cax.set_title(lab)
        cax.plot(df_orig['Date'], df_orig[lab], mrk, label='Originale')
        cax.plot(df_mod['Date'], df_mod[lab], label='Processato')
        cax.grid()
        cax.legend()


if __name__ == '__main__':
    df = read_cell(FILE_NAME, ORIGINAL_DATA)
    plot_cell(df)
    df = read_cell('Cella_14.csv', ORIGINAL_DATA)
    plot_cell(df)
    plot_correlation_matrix(df)
    plot_comparison_cell('Cella_13.csv', dottedplot=False)
    plt.show()