""" 
This module contains the function to import data from the specified cell number. 

@author: Matteo Dalle Vedove (matteodv99tn@gmail.com)
"""

import pandas as pd
import matplotlib.pyplot as plt
from constants import ORIGINAL_DATA_PATH, PROCESSED_DATA_PATH
from os.path import join, isfile
from datetime import datetime

# All available labels
#
# "Date",
# "InterventoTermostatoAntigelo",
# "PercentualeAperturaValvolaMiscelatrice",
# "PercentualeVelocitaVentilatori",
# "PompaAvviamentiGiornalieri",
# "PompaGlicoleMarcia",
# "Postventilazione",
# "Preventilazione",
# "Raffreddamento",
# "SbrinamentoAcqua",
# "SbrinamentoAcquaAvviamenti",
# "SbrinamentoAria",
# "SelettoreFrigoManuale",
# "SgocciolamentoDopoSbrinamentoAcqua",
# "TemperaturaCellaMediata",
# "TemperaturaCelle",
# "TemperaturaMandataGlicole",
# "TemperaturaMandataGlicoleNominale",
# "TemperaturaMele",
# "TemperaturaRitornoGlicole",
# "TemperaturaRoccia1",
# "TemperaturaRoccia2",
# "TemperaturaRoccia3",
# "TemperaturaRoccia4",
# "Umidificazione",
# "UmidificazioneAvviamenti",
# "UmiditaRelativa",
# "ValvolaSbrinamentoAcquaAperta",
# "VentilatoreAvviamentiGiornalieri",
# "VentilatoreMarcia",
# "VentilazioneAntistratificazionePortaAperta",
# "VentilazioneForzata"

ITALIAN_TO_ENGLISH = {
    "PercentualeAperturaValvolaMiscelatrice":   "mixing_valve_percentage",
    "PercentualeVelocitaVentilatori":           "ventilators_speed_percentage",
    "PompaGlicoleMarcia":                       "pump_status",
    "Raffreddamento":                           "cooling",
    "TemperaturaCelle":                         "cell_temperature",
    "TemperaturaCellaMediata":                  "mean_cell_temperature",
    "TemperaturaMandataGlicole":                "inlet_fluid_temperature",
    "TemperaturaMandataGlicoleNominale":        "inlet_setpoint_temperature",
    "TemperaturaRitornoGlicole":                "outlet_fluid_temperature",
    "TemperaturaMele":                          "apple_temperature",
    "TemperaturaRoccia1":                       "rock1_temperature",
    "TemperaturaRoccia2":                       "rock2_temperature",
    "TemperaturaRoccia3":                       "rock3_temperature",
    "TemperaturaRoccia4":                       "rock4_temperature",
    "Preventilazione":                          "pre_ventilation",
    "Postventilazione":                         "post_ventilation",
    "UmiditaRelativa":                          "humidity_percentage",
    "VentilatoreMarcia":                        "ventilation_status",
    "Minuti":                                   "minutes",
    "Date":                                     "date"  
}

ENGLISH_TO_ITALIAN = dict((v, k) for k, v in ITALIAN_TO_ENGLISH.items())
LABELS_TO_KEEP     = list(ITALIAN_TO_ENGLISH.keys())


def translate_to_english(df: pd.DataFrame) -> pd.DataFrame:
    """ Translates the column names of the specified dataframe from Italian to English. """
    new_names = list()
    for col_name in df.columns:
        if col_name in ITALIAN_TO_ENGLISH:
            new_names.append(ITALIAN_TO_ENGLISH[col_name])
        else:
            new_names.append(col_name)
    df.columns = new_names
    return df


def translate_to_italian(df: pd.DataFrame) -> pd.DataFrame:
    """ Translates the column names of the specified dataframe from English to Italian. """
    new_names = list()
    for col_name in df.columns:
        if col_name in ENGLISH_TO_ITALIAN:
            new_names.append(ENGLISH_TO_ITALIAN[col_name])
        else:
            new_names.append(col_name)
    df.columns = new_names
    return df


def import_data(cell_number:            int,
                processed_data:         bool = True,
                english_translation:    bool = False) -> pd.DataFrame:
    """ Imports data from the specified cell number. Furthermore, it adds a new column "Minuti" 
    which contains the time (in minutes) from the first measurement.

    Args:
        cell_number (int):                      The cell number to import data from.
        processed_data (bool, optional):        Whether to import processed data or not. 
                                                Defaults to True.
        english_translation (bool, optional):   Whether to translate the column names to English or 
                                                not.

    Returns:
        pd.DataFrame: The imported data.
    """

    if processed_data:
        base_path = PROCESSED_DATA_PATH
    else:
        base_path = ORIGINAL_DATA_PATH
    file_path = join(base_path, f"Cella_{cell_number}.csv")

    if(not isfile(file_path)):
        raise FileNotFoundError(
            f"File \"Cella_{cell_number}.csv\" do not exists in \"{base_path}\"")

    print("Loading", file_path)

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.dropna()
    df = df[df.columns.intersection(LABELS_TO_KEEP)]
    df = df.reset_index(drop=True)
    df["Minuti"] = (df["Date"] - df.Date[0]
                    ).apply(lambda x: x.total_seconds() / 60)

    if english_translation:
        df = translate_to_english(df)

    return df


def show_cell_data(df: pd.DataFrame,
                   begin: datetime = None,
                   end: datetime = None,
                   size: tuple = (15, 10)):
    """ Creates a plot of the data of the specified cell.

    Args:
        df (pd.DataFrame): dataframe containing the data to show.
        begin (datetime, optional): the start date. If not set, it shows starting from the first 
            entry in the dataframe.
        end (datetime, optional): the end date. If not set, is ends 4 days after the beginning.
        size (tuple, optional): the size of the plot. Defaults to (15, 10).
    """
    df = translate_to_italian(df)
    if begin is None:
        begin = df.Date[0]
    if end is None:
        end = begin + pd.Timedelta(days=4)

    df = df[(df.Date >= begin) & (df.Date <= end)]

    fig, ax = plt.subplots(2, 1, figsize=size, sharex=True)

    ax[0].plot(df.Date, df.TemperaturaMandataGlicole, label="inlet")
    ax[0].plot(df.Date, df.TemperaturaRitornoGlicole, label="outlet")
    ax[0].plot(df.Date, df.TemperaturaCelle,          label="cell")
    ax[0].plot(df.Date, df.TemperaturaMandataGlicoleNominale, "--", label="setpoint")
    ax[0].set_xlabel(None)
    ax[0].set_ylabel("Temperatures [°C]")
    ax[0].legend(loc="lower right")
    ax[0].grid(which="major", linestyle="dashed",
               linewidth="0.5", color="lightgrey")

    ax[1].plot(df.Date, 100*df.PompaGlicoleMarcia,
               label="pump status [on/off]")
    ax[1].plot(df.Date, df.PercentualeAperturaValvolaMiscelatrice,
               label="mixing valve aperture [%]")
    ax[1].grid(which="major", linestyle="dashed",
               linewidth="0.5", color="lightgrey")
    ax[1].legend(loc="lower right")
    ax[1].set_xlabel("Date")

    return fig


if __name__ == "__main__":
    df = import_data(13, False)
    show_cell_data(df, begin=datetime(2022, 10, 1))
    plt.show()
