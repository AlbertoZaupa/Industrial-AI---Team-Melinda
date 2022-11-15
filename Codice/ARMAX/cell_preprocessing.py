import  pandas              as pd
import  numpy               as np
import  matplotlib.pyplot   as plt
import  filter_shapes       as fs
import  os
from    pathlib             import Path

FILE_DIR        = str(os.path.realpath(os.path.dirname(__file__)))
tmp             = FILE_DIR[:FILE_DIR.rfind(os.sep)]
ROOT_DIR        = tmp[:tmp.rfind(os.sep)]
CSV_DIR         = os.path.join(ROOT_DIR, 'CSV')

COMPARISON_FILE = 'Cella_13.csv'
EXPORT_ALL      = False
SHOW_FILTERS    = True
COMPARE_ONLY    = False
""" 
    WEIGHTED MOVING AVERAGE NON-CAUSAL FILTER

Qua sotto è possibile definire i pesi per un filtro moving-average (AKA running-mean). E' sufficiente definire "mezzo"
filtro; l'idea è che all'indice "i" del vettore/lista è associata il peso dell' "i"esimo lag avanti nel tempo. 
Internamente la funzione specchia questo valore per considerare la stessa quantità "indietro nel tempo".
"""
# Definizione dei filtri
w_mandata   = fs.gaussian(10, 25)
w_ritorno   = fs.gaussian(10, 25)
w_nominale  = 3*fs.gaussian(40, 70) + fs.gaussian(6, 70)
w_celle     = fs.gaussian(15, 50) + fs.gaussian(4, 50)




def build_filter(half_weight: list) -> list:
    if type(half_weight) == np.ndarray:
        half_weight = half_weight.tolist()
    w0 = half_weight[1:]
    w0.reverse()
    return w0 + half_weight


def weighted_moving_average(
        data: pd.Series,
        weight: list
        ) -> pd.Series:
    w = build_filter(weight)
    out = pd.Series(np.zeros(len(w)))
    w0_index = int((len(w)-1)/2)

    for i in range(len(data)):
        tmp_sum = 0
        tmp_weight = 0
        for j in range(len(w)):
            try:
                tmp_sum += w[j]*data[i-w0_index+j]
                tmp_weight += w[j]
            except:
                pass
        out[i] = tmp_sum / tmp_weight

    return out


def plot_filter(weights, axis, figtitle=''):
    w = build_filter(weights)
    f_len = len(w)
    max_ind = int((f_len-1)/2)
    indexes = np.linspace(-max_ind, max_ind, f_len).tolist()
    axis.stem(indexes, w, label=figtitle)
    axis.legend()

def show_filters():
    global w_mandata  
    global w_ritorno  
    global w_nominale 
    global w_celle   
    f, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 5))
    plot_filter(w_mandata,      ax[0, 0], "Mandata")
    plot_filter(w_ritorno,      ax[0, 1], "Ritorno")
    plot_filter(w_nominale,     ax[1, 0], "Nominale")
    plot_filter(w_celle,        ax[1, 1], "Celle")
    plt.show()


def process_cell(filename: str) -> pd.DataFrame:
    path = os.path.join(CSV_DIR, 'october', filename)
    print('Processing', path)
    df = pd.read_csv(path, parse_dates=['Date'])
    df.dropna()

    timedelta = df.Date - df.Date[0]
    df['Minuti'] = timedelta.apply(lambda X: X.total_seconds() / 60)

    global w_mandata  
    global w_ritorno  
    global w_nominale 
    global w_celle     

    df['TemperaturaMandataGlicole'] = weighted_moving_average(
        df.TemperaturaMandataGlicole,
        w_mandata
        )
    df['TemperaturaRitornoGlicole'] = weighted_moving_average(
        df.TemperaturaRitornoGlicole,
        w_ritorno
        )
    df['TemperaturaMandataGlicoleNominale'] = weighted_moving_average(
        df.TemperaturaMandataGlicoleNominale,
        w_nominale
        )
    df['TemperaturaCelle'] = weighted_moving_average(
        df.TemperaturaCelle,
        w_celle
        )

    Path(os.path.join(CSV_DIR, 'modified')).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(CSV_DIR, 'modified', filename), index=False)

    return df


def compare_cell(filename: str) -> None:
    df_orig = pd.read_csv(os.path.join(CSV_DIR, 'october', filename))
    df_mod = pd.read_csv(os.path.join(CSV_DIR, 'modified', filename))

    compare_labels = [
        'TemperaturaMandataGlicole',
        'TemperaturaMandataGlicoleNominale',
        'TemperaturaRitornoGlicole',
        'TemperaturaCelle'
        ]

    for idx, lab in enumerate(compare_labels):
        plt.subplot(4, 1, idx+1)
        plt.title(lab)
        plt.plot(df_orig[lab], label='Originale')
        plt.plot(df_mod[lab], label='Modificato')
        plt.legend()
    plt.show()


if SHOW_FILTERS:
    show_filters()

if COMPARE_ONLY:
    compare_cell(COMPARISON_FILE)
    exit(0)

if EXPORT_ALL:
    print('Processing all cells')
    filelist = os.listdir(os.path.join(CSV_DIR, 'october'))
    for filename in filelist:
        process_cell(filename)
else:
    process_cell(COMPARISON_FILE)
    compare_cell(COMPARISON_FILE)