from os.path import dirname, abspath, join, normpath

PATH                = dirname(abspath(__file__))
ROOT_PATH           = normpath(join(PATH, '..', '..'))
ORIGINAL_DATA_PATH  = join(ROOT_PATH, 'CSV', 'Original data')
PROCESSED_DATA_PATH = join(ROOT_PATH, 'CSV', 'Processed data')
SYSID_PATH          = join(ROOT_PATH, 'Codice', 'System Identification')

if __name__ == '__main__':
    print('%30s:'%'current file directory', PATH)
    print('%30s:'%'project root directory', ROOT_PATH)
    print('%30s:'%'original CSV data location', ORIGINAL_DATA_PATH)
    print('%30s:'%'processed CSV data location', PROCESSED_DATA_PATH)
    print('%30s:'%'System identification code', SYSID_PATH)