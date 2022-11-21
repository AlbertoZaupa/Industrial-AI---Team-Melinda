# Struttura della repository
Nella cartella [_Note_](Note/) è possibile trovare appunti condivisi per capire vari elementi come:
  - [lo studio degli scambiatori di calore](Note/scambio_termico.md)
  - [**variabili e informazioni sul dataset**](Note/dataset.md)
  
Dentro la cartella [Codice](Codice/) è possibile trovare tutte le implementazioni dei modelli proposti come soluzione, in particolare
- [ARMAX](Codice/ARMAX/) usata per i modelli _Auto-Regressive with eXogenous elements_:
  - [jupyter notebook](Codice/ARMAX/ARX.ipynb) che carica i dati dalle celle ed effettua una regressione lineare. Per funzionare necessita di:
  - [codegeneration.py](Codice/ARMAX/codegeneration.py) e [auto_generated_code.py](Codice/ARMAX/auto_generated_code.py) che vengono utilizzati in accoppiata per semplificare la definizione dei diversi modelli lineari.
  - [cell_preprocessing.py](Codice/ARMAX/cell_preprocessing.py) e [import_dataframe.py](Codice/ARMAX/import_dataframe.py) sono i file che vengono utilizzati per fare un pre-filtraggio sui dati e definiscono delle funzioni che vengono chiamati nel notebook jupiter.
- [RNNs(Keras)](Codice/RNNs(Keras)) contenente il codice per le reti neurali utilizzate:
  - *TODO*


# Reinforcement Learning

### Guida al codice

Il codice dell'agente di RL si trova in [Codice/RL/](). I file all'interno della cartella sono
i seguenti:
- [main.py](Codice/RL/main.py): script contenente il loop di allenamento.
- [config.py](Codice/RL/config.py): variabili di configurazione.
- [environment.py](Codice/RL/environment.py): codice per simulazione del comportamento della cella frigorifera.
- [agent.py](Codice/RL/agent.py): classe `Agent`, wrapper delle reti neurali actor e critic.
- [reward.py](Codice/RL/reward.py): definizione delle funzioni di ricompensa.
- [buffer.py](Codice/RL/buffer.py): implementazione del replay buffer.

Prima di eseguire il codice è necessario configurare alcuni parametri secondo il proprio ambiente di lavoro.
In particolare è necessario specificare il percorso dove si trovano i file corrispondenti alle reti neurali utilizzate per la simulazione, e la directory in cui verrà salvata la rete neurale corrispondente all'agente.

Per allenare l'agente:
```
$ cd ./Codice/RL

$ pip3 install -r requirements.txt

$ python3 main.py
```

Al termine dell'allenamento, che su MacBookAir M1 impiega circa 20 minuti con i parametri attuali, all'utente viene mostrato un prompt che chiede se si vuole salvare l'agente su disco. 

### Introduzione
Essendo stati in grado di ottenere dei modelli predittivi in generale abbastanza efficaci,
abbiamo voluto sfruttarli per cercare di scoprire la legge ottimale secondo cui impostare la 
temperatura del glicole che viene mandato alla cella. 

In particolare abbiamo scelto di utilizzare delle
tecniche di Reinforcement Learning, il cui obbiettivo è quello di produrre un'agente (ovvero una funzione)
che ha appreso la legge ottimale `µ(s)` ('s' è lo stato della cella). La legge ottimale che l'agente impara è
strettamente legata alla funzione di ricompensa `r(a, s)` ('a' è l'azione scelta dall'agente), che deve di 
conseguenza essere definita in modo appropriato.

### Simulazione

Le tecniche utilizzate prevedono che l'agente che cerchiamo di allenare possa interagire con l'ambiente
in cui opera, eseguendo delle azioni e collezionando ricompense e penalità che ne derivano.

Utilizzando i modelli predittivi prima citati, abbiamo simulato il comportamento di una cella frigorifera
in base alle azioni scelte dall'agente.

### Training loop

Il processo di allenamento dell'agente è costituito da una successione di episodi, il cui numero e
durata temporale all'interno della simulazione sono definiti dall'utente.
All'inizio di ogni episodio lo stato della cella viene inzializzato con un campione di alcune ore,
proveniente dal dataset a disposizione. 

Fino al termine dell'episodio di allenamento, l'agente ad ogni passo sceglie la temperatura a cui impostare
il glicole ed osserva la ricompensa che proviene dall'ambiente, salvando l'osservazione in un "replay buffer".
Prima di far avanzare la simulazione dell'episodio di un'unità di tempo, vengono aggiornati i pesi delle reti neurali dell'agente (paragrafo successivo).

### Algoritmo scelto
Abbiamo scelto di implementare l'algoritmo DDPG ([link al paper](https://arxiv.org/abs/1509.02971)). L'agente
quindi è costituito da una coppia di reti neurali, `actorNN(s)`, e `criticNN(s, a)`. La prima sceglie che 
azione compiere dato lo stato corrente del sistema, mentre la seconda giudica il valore a lungo termine dell'
azione scelta.

### Shallow weight updates
Secondo comune pratica nell'ambito del Deep RL, le due reti che costituiscono l'agente sono affiancate altre
due reti equivalenti `target_actorNN(s)` e `target_criticNN(s, a)`, utilizzate nell'equazione di Bellman per 
migliorare le proprietà di convergenza e stabilità numerica del processo di allenamento. Si tratta della tecnica spesso chiamata "shallow weight updates".

### Funzione di ricompensa
Definire una funzione di ricompensa adeguata è un passaggio fondamentale, e che può richiedere diverse 
iterazioni. Abbiamo scelto di definire una funzione di ricompensa molto semplice: l'agente ottiene un
feedback positivo per ogni passo della simulazione durante il quale la pompa rimane spenta, mentre quando
questa è accesa l'agente riceve un feedback negativo proporzionale alla temperatura del glicole, dunque
la penalità è tanto maggiore quanto più la temperatura è bassa.

### Possibili sviluppi futuri

Pensiamo che una prospettiva molto promettente sia quella di, una volta che la sensoristica sarà presente,
utilizzare direttamente il consumo di energia derivante dalla produzione del freddo come feedback per l'agente.

Nel 2016 Google ha pubblicato un [paper](https://static.googleusercontent.com/media/research.google.com/it//pubs/archive/42542.pdf) in cui viene discussa l'implementazione di una rete neurale in grado di comprendere il consumo di energia del datacenter in base ad una serie di parametri di controllo. Nei due anni successivi, in collaborazione con DeepMind, questa rete è stata utilizzata per implementare un algoritmo RL capace di amministrare il datacenter in modo efficente.

Un'altra prospettiva interessante potrebbe essere applicare un algoritmo simile a tutto l'ipogeo, dando la possibilità all'agente di controllare tutte le celle contemporaneamente. Pensiamo che questo approccio, combinato con l'utilizzo del consumo energetico come feedback per l'agente, possa portare risultati molto interessanti.
