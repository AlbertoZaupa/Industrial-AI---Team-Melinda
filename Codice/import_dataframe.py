import pandas as pd
from constants import DATA_DIR
import os

def get_cell_list() -> tuple:
    # return tuple(os.listdir(DATA_DIR))
    return ('Ipogeo2_Cella13.csv', 'Ipogeo2_Cella14.csv')

s = 'ciao_'
s.find('_')


def import_csv() -> dict:
    
    res = {}

    for filename in get_cell_list():
        print('Importing', filename)
        lbl = filename[filename.find('_')+1:filename.find('.')]
        df = pd.read_csv(
                os.path.join(DATA_DIR, filename),
                parse_dates=['Date']
            )
        df.drop(df.columns[0], axis=1, inplace=True)
        
        col_names = ['Date']
        for i in range(1, len(df.columns)):
            c = df.columns[i]
            col_names.append( c[len(filename)-4:] )
        df.columns = col_names

        res[lbl] = df
    
    return res



if __name__ == '__main__':
    from os import system
    system('cls')
    cells = import_csv()

    for e in cells:
        print(e)

    print(type(cells['Cella13'].TemperaturaCelle[1] ))