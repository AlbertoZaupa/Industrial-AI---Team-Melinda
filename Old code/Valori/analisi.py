import os
import re
import pandas as pd
import numpy as np

# Questo script prende i dati da un file csv e genera un csv in output con diversi
# parametri come la percentuale massima dell'apertura della valvola, 
# la media dell'apertura della valvola, ecc.

NUMERO_CELLE = 34
FILE_NAME = "giugno_15min.csv"
INPUT_FULL_PATH = os.path.join("15min", FILE_NAME)
OUTPUT_FULL_PATH = os.path.join("15min", f"riassunto_{FILE_NAME}")

# Definisco dei range di temperatura
# Se un determinato dato va oltre questi range, il dato viene considerato non valido e viene ignorato
RANGE_TEMPERATURA_GLICOLE = (-10, 10)
RANGE_TEMPERATURA_MELE = (-2, 5)
RANGE_TEMPERATURA_CELLA = (-3, 6)

df = pd.read_csv(INPUT_FULL_PATH, sep=";", low_memory=False)

output_data = {
    "Date": df["Date"].tolist(),
    "Numero cella percentuale massima apertura valvola": [],
    "Percentuale massima apertura valvola": [],
    "Media percentuale apertura valvola": [],
    "Deviazione standard percentuale apertura valvola": [],
    "75esimo percentile apertura valvole": [],
    "90esimo percentile apertura valvole": [],
    "Numero celle con valvole aperte": [],
    "Numero celle con ventilazione accesa": [],
    "Numero cella massimo avviamenti giornalieri pompa": [],
    "Massimo avviamenti giornalieri pompa": [],
    "Numero cella massimo avviamenti giornalieri ventilatore": [],
    "Massimo avviamenti giornalieri ventilatore": [],
    "Numero cella temperatura minima (aria)": [],
    "Temperatura minima celle (aria)": [],
    "Numero cella temperatura massima (aria)": [],
    "Temperatura massima celle (aria)": [],
    "Media temperatura celle (aria)": [],
    "Deviazione standard temperatura celle (aria)": [],
    "Numero cella temperatura minima (mele)": [],
    "Temperatura minima celle (mele)": [],
    "Numero cella temperatura massima (mele)": [],
    "Temperatura massima celle (mele)": [],
    "Media temperatura celle (mele)": [],
    "Deviazione standard temperatura celle (mele)": [],
    "Numero cella temperatura minima (uscita ventilatore)": [],
    "Temperatura minima celle (uscita ventilatore)": [],
    "Numero cella temperatura massima (uscita ventilatore)": [],
    "Temperatura massima celle (uscita ventilatore)": [],
    "Media temperatura celle (uscita ventilatore)": [],
    "Deviazione standard temperatura celle (uscita ventilatore)": [],
    "Numero cella max delta temperatura glicole": [],
    "Temperatura mandata glicole con min delta": [],
    "Max delta temperatura glicole": [],
    "Numero cella min delta temperatura glicole": [],
    "Temperatura mandata glicole con max delta": [],
    "Min delta temperatura glicole": [],
}


def field2float(value):
    """
    Siccome i campi di una cella del csv sono misti (possono essere int64 o str)
    converto i numeri di una cella in float
    Da notare che i float sono numeri con parte intera e
    decimale delimitata da una virgola e non un punto
    """
    if pd.isnull(value):
        return value
    elif isinstance(value, str):
        return float(value.replace(",", "."))
    else:
        return float(value)


def field2int(value):
    if pd.isnull(value):
        return value
    else:
        return int(value)


def float2field(value):
    if value is None:
        return value
    else:
        return str(value).replace(".", ",")


def int2field(value):
    if value is None:
        return value
    else:
        return str(value)


def extract_numero_cella(s):
    """
    Da una stringa di un campo del dataset estrae il numero della cella corrispondente
    """
    cella = re.search(r"Cella [0-9]+", s).group(0)
    return int(re.search(r"[0-9]+", cella).group(0))


def find_dati_valvole(df: pd.DataFrame, i: int):
    """
    Trova in una riga del dataset la cella con la valvola piu aperta,
    di quanto è aperta la valvola, la media e la deviazione standard dell'apertura delle valvole
    """

    numero_cella_max_apertura_valvola = -1
    max_percentuale_apertura_valvola = 0.0
    valori_percentuale_apertura_valvola = []

    k: str
    for k in df.filter(regex=r"(Cella [0-9]+).*(Valvola Miscelatrice)"):
        valore_valvola = field2float(df[k][i])
        if pd.isnull(valore_valvola):
            continue
        valori_percentuale_apertura_valvola.append(valore_valvola)
        if max_percentuale_apertura_valvola < valore_valvola:
            max_percentuale_apertura_valvola = valore_valvola
            numero_cella_max_apertura_valvola = extract_numero_cella(k)

    if valori_percentuale_apertura_valvola:
        return (
            numero_cella_max_apertura_valvola,
            max_percentuale_apertura_valvola,
            np.mean(valori_percentuale_apertura_valvola),
            np.std(valori_percentuale_apertura_valvola),
            np.percentile(valori_percentuale_apertura_valvola, 75),
            np.percentile(valori_percentuale_apertura_valvola, 90),
        )
    else:
        return (None,) * 6


def find_dati_valvole_aperte_e_ventilazione_accesa(df: pd.DataFrame, i: int):
    """
    Trova in una riga del dataset quali celle hanno valvole aperte e ventilazione accesa.
    Vengono controllati i campi "Marcia Pompa Glicole" e "Marcia Ventilatore"
    """

    celle_con_valvole_aperte = 0
    for k in df.filter(regex=r"(Cella [0-9]+).*(Marcia Pompa Glicole)"):
        valore_valvola = field2float(df[k][i])
        if pd.isnull(valore_valvola):
            continue
        if valore_valvola != 0:
            if celle_con_valvole_aperte is None:
                celle_con_valvole_aperte = 1
            else:
                celle_con_valvole_aperte += 1
        elif celle_con_valvole_aperte is None:
            celle_con_valvole_aperte = 0

    celle_con_ventilazione_accesa = 0
    for k in df.filter(regex=r"(Cella [0-9]+).*(Marcia Ventilatore)"):
        valore_ventilazione = field2float(df[k][i])
        if pd.isnull(valore_ventilazione):
            continue
        if valore_ventilazione != 0:
            if celle_con_ventilazione_accesa is None:
                celle_con_ventilazione_accesa = 1
            else:
                celle_con_ventilazione_accesa += 1
        elif celle_con_ventilazione_accesa is None:
            celle_con_ventilazione_accesa = 0

    return (celle_con_valvole_aperte, celle_con_ventilazione_accesa)


def find_avviamenti_giornalieri(df: pd.DataFrame, i: int):
    """
    Trova il numero di avviamenti giornalieri della pompa e del ventilatore
    """

    numero_cella_max_avviamenti_giornalieri_pompa = None
    max_avviamenti_giornalieri_pompa = None
    numero_cella_max_avviamenti_giornalieri_ventilatore = None
    max_avviamenti_giornalieri_ventilatore = None

    for k in df.filter(regex=r"(Cella [0-9]+).*(Avviamenti Giornalieri Pompa)"):
        avviamenti_giornalieri_pompa = field2int(df[k][i])
        if pd.isnull(avviamenti_giornalieri_pompa):
            continue
        if (
            max_avviamenti_giornalieri_pompa is None
            or max_avviamenti_giornalieri_pompa < avviamenti_giornalieri_pompa
        ):
            max_avviamenti_giornalieri_pompa = avviamenti_giornalieri_pompa
            numero_cella_max_avviamenti_giornalieri_pompa = extract_numero_cella(k)

    for k in df.filter(regex=r"(Cella [0-9]+).*(Avviamenti Giornalieri Ventilatore)"):
        avviamenti_giornalieri_ventilatore = field2int(df[k][i])
        if pd.isnull(avviamenti_giornalieri_ventilatore):
            continue
        if (
            max_avviamenti_giornalieri_ventilatore is None
            or max_avviamenti_giornalieri_ventilatore
            < avviamenti_giornalieri_ventilatore
        ):
            max_avviamenti_giornalieri_ventilatore = avviamenti_giornalieri_ventilatore
            numero_cella_max_avviamenti_giornalieri_ventilatore = extract_numero_cella(
                k
            )

    return (
        numero_cella_max_avviamenti_giornalieri_pompa,
        max_avviamenti_giornalieri_pompa,
        numero_cella_max_avviamenti_giornalieri_ventilatore,
        max_avviamenti_giornalieri_ventilatore,
    )


def find_dati_temperature(df: pd.DataFrame, i: int):
    """
    Trova i dati dei diversi tipi di temperature,
    min e max aria, min e max mele, min e max ventilatore
    """

    numero_cella_min_temperatura_aria = None
    min_temperatura_aria = None
    numero_cella_max_temperatura_aria = None
    max_temperatura_aria = None
    valori_temperatura_aria = []
    valori_aria = None

    numero_cella_min_temperatura_mele = None
    min_temperatura_mele = None
    numero_cella_max_temperatura_mele = None
    max_temperatura_mele = None
    valori_temperatura_mele = []
    valori_mele = None

    numero_cella_min_temperatura_ventilatore = None
    min_temperatura_ventilatore = None
    numero_cella_max_temperatura_ventilatore = None
    max_temperatura_ventilatore = None
    valori_temperatura_ventilatore = []
    valori_ventilatore = None

    for k in df.filter(regex=r"(Cella [0-9]+).*(Temperatura Cella)$"):
        valore_temperatura = field2float(df[k][i])
        if pd.isnull(valore_temperatura):
            continue

        if (
            valore_temperatura < RANGE_TEMPERATURA_CELLA[0]
            or valore_temperatura > RANGE_TEMPERATURA_CELLA[1]
        ):
            continue

        numero_cella = extract_numero_cella(k)
        valori_temperatura_aria.append(valore_temperatura)

        if min_temperatura_aria is None or min_temperatura_aria > valore_temperatura:
            min_temperatura_aria = valore_temperatura
            numero_cella_min_temperatura_aria = numero_cella
        if max_temperatura_aria is None or max_temperatura_aria < valore_temperatura:
            max_temperatura_aria = valore_temperatura
            numero_cella_max_temperatura_aria = numero_cella

    for k in df.filter(regex=r"(Cella [0-9]+).*(Temperatura Mele)$"):
        valore_temperatura = field2float(df[k][i])
        if pd.isnull(valore_temperatura):
            continue

        if (
            valore_temperatura < RANGE_TEMPERATURA_MELE[0]
            or valore_temperatura > RANGE_TEMPERATURA_MELE[1]
        ):
            continue

        numero_cella = extract_numero_cella(k)
        valori_temperatura_mele.append(valore_temperatura)

        if min_temperatura_mele is None or min_temperatura_mele > valore_temperatura:
            min_temperatura_mele = valore_temperatura
            numero_cella_min_temperatura_mele = numero_cella
        if max_temperatura_mele is None or max_temperatura_mele < valore_temperatura:
            max_temperatura_mele = valore_temperatura
            numero_cella_max_temperatura_mele = numero_cella

    for k in df.filter(regex=r"(Cella [0-9]+).*(Temperatura Mandata Glicole)$"):
        valore_temperatura = field2float(df[k][i])
        if pd.isnull(valore_temperatura):
            continue

        if (
            valore_temperatura < RANGE_TEMPERATURA_CELLA[0]
            or valore_temperatura > RANGE_TEMPERATURA_CELLA[1]
        ):
            continue

        numero_cella = extract_numero_cella(k)
        valori_temperatura_ventilatore.append(valore_temperatura)

        if (
            min_temperatura_ventilatore is None
            or min_temperatura_ventilatore > valore_temperatura
        ):
            min_temperatura_ventilatore = valore_temperatura
            numero_cella_min_temperatura_ventilatore = numero_cella
        if (
            max_temperatura_ventilatore is None
            or max_temperatura_ventilatore < valore_temperatura
        ):
            max_temperatura_ventilatore = valore_temperatura
            numero_cella_max_temperatura_ventilatore = numero_cella

    if valori_temperatura_aria:
        valori_aria = (
            numero_cella_min_temperatura_aria,
            min_temperatura_aria,
            numero_cella_max_temperatura_aria,
            max_temperatura_aria,
            np.mean(valori_temperatura_aria),
            np.std(valori_temperatura_aria),
        )
    else:
        valori_aria = (None,) * 6

    if valori_temperatura_mele:
        valori_mele = (
            numero_cella_min_temperatura_mele,
            min_temperatura_mele,
            numero_cella_max_temperatura_mele,
            max_temperatura_mele,
            np.mean(valori_temperatura_mele),
            np.std(valori_temperatura_mele),
        )
    else:
        valori_mele = (None,) * 6

    if valori_temperatura_ventilatore:
        valori_ventilatore = (
            numero_cella_min_temperatura_ventilatore,
            min_temperatura_ventilatore,
            numero_cella_max_temperatura_ventilatore,
            max_temperatura_ventilatore,
            np.mean(valori_temperatura_ventilatore),
            np.std(valori_temperatura_ventilatore),
        )
    else:
        valori_ventilatore = (None,) * 6

    return valori_aria + valori_mele + valori_ventilatore


def find_delta_temperatura_glicole(df: pd.DataFrame, i: int):
    """
    Trova il delta tra "Temperatura Mandata Glicole" e "Temperatura Ritorno Glicole"
    per vedere quale cella assorbe più calore
    """

    numero_cella_max_delta_temperatura_glicole = None
    temperatura_glicole_min_delta = None
    max_delta_temperatura_glicole = None
    numero_cella_min_delta_temperatura_glicole = None
    temperatura_glicole_max_delta = None
    min_delta_temperatura_glicole = None

    for k_mandata, k_ritorno in zip(
        df.filter(regex=r"(Cella [0-9]+).*(Temperatura Mandata Glicole)$"),
        df.filter(regex=r"(Cella [0-9]+).*(Temperatura Ritorno Glicole)$"),
    ):
        valore_temperatura_mandata = field2float(df[k_mandata][i])
        valore_temperatura_ritorno = field2float(df[k_ritorno][i])
        if pd.isnull(valore_temperatura_mandata) or pd.isnull(
            valore_temperatura_ritorno
        ):
            continue

        if (
            valore_temperatura_mandata < RANGE_TEMPERATURA_CELLA[0]
            or valore_temperatura_mandata > RANGE_TEMPERATURA_CELLA[1]
            or valore_temperatura_ritorno < RANGE_TEMPERATURA_CELLA[0]
            or valore_temperatura_ritorno > RANGE_TEMPERATURA_CELLA[1]
        ):
            continue

        numero_cella = extract_numero_cella(k_ritorno)
        delta_temperatura = abs(valore_temperatura_mandata - valore_temperatura_ritorno)

        if (
            min_delta_temperatura_glicole is None
            or min_delta_temperatura_glicole > delta_temperatura
        ):
            min_delta_temperatura_glicole = delta_temperatura
            temperatura_glicole_min_delta = valore_temperatura_mandata
            numero_cella_min_delta_temperatura_glicole = numero_cella
        if (
            max_delta_temperatura_glicole is None
            or max_delta_temperatura_glicole < delta_temperatura
        ):
            max_delta_temperatura_glicole = delta_temperatura
            temperatura_glicole_max_delta = valore_temperatura_mandata
            numero_cella_max_delta_temperatura_glicole = numero_cella

    return (
        numero_cella_max_delta_temperatura_glicole,
        temperatura_glicole_min_delta,
        max_delta_temperatura_glicole,
        numero_cella_min_delta_temperatura_glicole,
        temperatura_glicole_max_delta,
        min_delta_temperatura_glicole,
    )


def update_output_data():
    # Itero per trovare per ogni riga le celle con la valvola piu
    # aperta e il valore percentuale di apertura della valvola
    for i in range(len(df)):
        (
            numero_cella_max_apertura_valvola,
            max_percentuale_apertura_valvola,
            media_percentuale_apertura_valvola,
            deviazione_standard_percentuale_apertura_valvola,
            percentile_75_apertura_valvole,
            percentile_90_apertura_valvole,
        ) = find_dati_valvole(df, i)
        output_data["Numero cella percentuale massima apertura valvola"].append(
            int2field(numero_cella_max_apertura_valvola)
        )
        output_data["Percentuale massima apertura valvola"].append(
            float2field(max_percentuale_apertura_valvola)
        )
        output_data["Media percentuale apertura valvola"].append(
            float2field(media_percentuale_apertura_valvola)
        )
        output_data["Deviazione standard percentuale apertura valvola"].append(
            float2field(deviazione_standard_percentuale_apertura_valvola)
        )
        output_data["75esimo percentile apertura valvole"].append(
            float2field(percentile_75_apertura_valvole)
        )
        output_data["90esimo percentile apertura valvole"].append(
            float2field(percentile_90_apertura_valvole)
        )

        (
            numero_celle_valvole_aperte,
            numero_celle_ventilazione_accesa,
        ) = find_dati_valvole_aperte_e_ventilazione_accesa(df, i)

        output_data["Numero celle con valvole aperte"].append(
            int2field(numero_celle_valvole_aperte)
        )
        output_data["Numero celle con ventilazione accesa"].append(
            int2field(numero_celle_ventilazione_accesa)
        )

        (
            numero_cella_avviamenti_giornalieri_pompa,
            massimo_avviamenti_giornalieri_pompa,
            numero_cella_avviamenti_giornalieri_ventilatore,
            massimo_avviamenti_giornalieri_ventilatore,
        ) = find_avviamenti_giornalieri(df, i)

        output_data["Numero cella massimo avviamenti giornalieri pompa"].append(
            int2field(numero_cella_avviamenti_giornalieri_pompa)
        )
        output_data["Massimo avviamenti giornalieri pompa"].append(
            int2field(massimo_avviamenti_giornalieri_pompa)
        )
        output_data["Numero cella massimo avviamenti giornalieri ventilatore"].append(
            int2field(numero_cella_avviamenti_giornalieri_ventilatore)
        )
        output_data["Massimo avviamenti giornalieri ventilatore"].append(
            int2field(massimo_avviamenti_giornalieri_ventilatore)
        )

        (
            numero_cella_min_temperatura_aria,
            min_temperatura_aria,
            numero_cella_max_temperatura_aria,
            max_temperatura_aria,
            media_temperatura_aria,
            deviazione_standard_temperatura_aria,
            numero_cella_min_temperatura_mele,
            min_temperatura_mele,
            numero_cella_max_temperatura_mele,
            max_temperatura_mele,
            media_temperatura_mele,
            deviazione_standard_temperatura_mele,
            numero_cella_min_temperatura_ventilatore,
            min_temperatura_ventilatore,
            numero_cella_max_temperatura_ventilatore,
            max_temperatura_ventilatore,
            media_temperatura_ventilatore,
            deviazione_standard_temperatura_ventilatore,
        ) = find_dati_temperature(df, i)

        output_data["Numero cella temperatura minima (aria)"].append(
            int2field(numero_cella_min_temperatura_aria)
        )
        output_data["Temperatura minima celle (aria)"].append(
            float2field(min_temperatura_aria)
        )
        output_data["Numero cella temperatura massima (aria)"].append(
            int2field(numero_cella_max_temperatura_aria)
        )
        output_data["Temperatura massima celle (aria)"].append(
            float2field(max_temperatura_aria)
        )
        output_data["Media temperatura celle (aria)"].append(
            float2field(media_temperatura_aria)
        )
        output_data["Deviazione standard temperatura celle (aria)"].append(
            float2field(deviazione_standard_temperatura_aria)
        )

        output_data["Numero cella temperatura minima (mele)"].append(
            int2field(numero_cella_min_temperatura_mele)
        )
        output_data["Temperatura minima celle (mele)"].append(
            float2field(min_temperatura_mele)
        )
        output_data["Numero cella temperatura massima (mele)"].append(
            int2field(numero_cella_max_temperatura_mele)
        )
        output_data["Temperatura massima celle (mele)"].append(
            float2field(max_temperatura_mele)
        )
        output_data["Media temperatura celle (mele)"].append(
            float2field(media_temperatura_mele)
        )
        output_data["Deviazione standard temperatura celle (mele)"].append(
            float2field(deviazione_standard_temperatura_mele)
        )

        output_data["Numero cella temperatura minima (uscita ventilatore)"].append(
            int2field(numero_cella_min_temperatura_ventilatore)
        )
        output_data["Temperatura minima celle (uscita ventilatore)"].append(
            float2field(min_temperatura_ventilatore)
        )
        output_data["Numero cella temperatura massima (uscita ventilatore)"].append(
            int2field(numero_cella_max_temperatura_ventilatore)
        )
        output_data["Temperatura massima celle (uscita ventilatore)"].append(
            float2field(max_temperatura_ventilatore)
        )
        output_data["Media temperatura celle (uscita ventilatore)"].append(
            float2field(media_temperatura_ventilatore)
        )
        output_data[
            "Deviazione standard temperatura celle (uscita ventilatore)"
        ].append(float2field(deviazione_standard_temperatura_ventilatore))

        (
            numero_cella_max_delta_temperatura_glicole,
            temperatura_glicole_min_delta,
            max_delta_temperatura_glicole,
            numero_cella_min_delta_temperatura_glicole,
            temperatura_glicole_max_delta,
            min_delta_temperatura_glicole,
        ) = find_delta_temperatura_glicole(df, i)

        output_data["Numero cella max delta temperatura glicole"].append(
            int2field(numero_cella_max_delta_temperatura_glicole)
        )
        output_data["Temperatura mandata glicole con min delta"].append(
            float2field(temperatura_glicole_min_delta)
        )
        output_data["Max delta temperatura glicole"].append(
            float2field(max_delta_temperatura_glicole)
        )
        output_data["Numero cella min delta temperatura glicole"].append(
            int2field(numero_cella_min_delta_temperatura_glicole)
        )
        output_data["Temperatura mandata glicole con max delta"].append(
            float2field(temperatura_glicole_max_delta)
        )
        output_data["Min delta temperatura glicole"].append(
            float2field(min_delta_temperatura_glicole)
        )

    # Scrivo in output csv
    pd.DataFrame(data=output_data).to_csv(OUTPUT_FULL_PATH, sep=";", index=False)


update_output_data()
