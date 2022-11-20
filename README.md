# Struttura della repository
- Nella cartella [_Note_](Note/) è possibile trovare appunti condivisi per capire vari elementi come:
  - [lo studio degli scambiatori di calore](Note/scambio_termico.md)
  - [**variabili e informazioni sul dataset**](Note/dataset.md)

Dentro la cartella [Codice](Codice/) è possibile trovare tutte le implementazioni dei modelli proposti come soluzione, in particolare
- [ARMAX](Codice/ARMAX/) usata per i modelli _Auto-Regressive with eXogenous elements_:
  - [jupyter notebook](Codice/ARMAX/ARX.ipynb) che carica i dati dalle celle ed effettua una regressione lineare. Per funzionare necessita di:
  - [codegeneration.py](Codice/ARMAX/codegeneration.py) e [auto_generated_code.py](Codice/ARMAX/auto_generated_code.py) che vengono utilizzati in accoppiata per semplificare la definizione dei diversi modelli lineari.
  - [cell_preprocessing.py](Codice/ARMAX/cell_preprocessing.py) e [import_dataframe.py](Codice/ARMAX/import_dataframe.py) sono i file che vengono utilizzati per fare un pre-filtraggio sui dati e definiscono delle funzioni che vengono chiamati nel notebook jupiter.