import  pandas      as pd
import  os.path
from    constants   import DATA_DIR
from    collections import defaultdict

def get_cell_list() -> tuple:
    ''' Ritorna il nome di tutti i file presenti nella cartella "CSV" '''
    return tuple(os.listdir(DATA_DIR))

def str_to_float(x):
    if type(x) == str:
        return float(x.replace(',', '.'))
    return x
    
def import_single_csv(filename: str) -> pd.DataFrame:
    ''' Dato il nome del file, genera il dataframe pandas della cella '''
    df = pd.read_csv(
            os.path.join(DATA_DIR, filename),
            parse_dates=['Date']            
        )
    df.drop( # rimuovo la prima colonna che è un duplicato degli indici
            df.columns[0], 
            axis=1, 
            inplace=True
        )
    # Rimuovo il prefisso della cella
    new_col_names = ['Date']
    for i in range(1, len(df.columns)):
        new_col_names.append( df.columns[i][len(filename)-4:])
    df.columns = new_col_names
    # Converto eventuali valori stringhe in floats
    df = df.applymap(str_to_float, )
    df.dropna()
    return df

def import_all_csv() -> defaultdict:
    ''' Crea un dizionario con tutti i file csv 
    L'idea è che dopo è possibile chiamare i dataframe di ogni singola cella 
    come `res['nome_lotto']['nome_cella']`, 
    per esempio `res['Ipogeo2']['Cella13']`.
    '''
    res = defaultdict(dict)
    for filename in get_cell_list():
        lbl_lot  = filename[:filename.find('_')]
        lbl_cell = filename[filename.find('_')+1:filename.find('.')]
        print(f'Importing file "{filename}"')
        res[lbl_lot][lbl_cell] = import_single_csv(filename)
    
    return res



if __name__ == '__main__':
    from os import system
    system('cls')
    cells = import_all_csv()
