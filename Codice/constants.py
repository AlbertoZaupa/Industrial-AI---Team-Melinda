import os

SOURCE_DIR    = os.path.realpath(os.path.dirname(__file__))
ROOT_DIR      = SOURCE_DIR[:SOURCE_DIR.rfind(os.sep)]
PURE_DATA_DIR = os.path.join(ROOT_DIR, 'Dati originali')
DATA_DIR      = os.path.join(ROOT_DIR, 'Dati')
