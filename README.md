# Struttura della repository
- Nella cartella [_Note_](Note/) è possibile trovare appunti condivisi per capire vari elementi come:
  - [lo studio degli scambiatori di calore](Note/scambio_termico.md)
  - [**variabili e informazioni sul dataset**](Note/dataset.md)

## Reinforcement Learning

### Guida al codice

Il codice dell'agente di RL si trova in `Codice/RL/`. I file all'interno della cartella sono
i seguenti:
- `main.py`: script contenente il loop di allenamento.
- `config.py`: variabili di configurazione.
- `environment.py`: codice per simulazione del comportamento della cella frigorifera.
- `agent.py`: classe `Agent`, wrapper delle reti neurali actor e critic.
- `reward.py`: definizione delle funzioni di ricompensa.
- `buffer.py`: implementazione del replay buffer.

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
in cui opera, eseguendo delle azioni e collezionando le ricompense che ne derivano, positive o negative
che siano.

Utilizzando i modelli predittivi prima citati, abbiamo simulato il comportamento di una cella frigorifera
in base alle azioni scelte dall'agente.

### Training loop

Il processo di allenamento dell'agente è costituito da una successione di episodi, il cui numero e
durata temporale all'interno della simulazione sono definiti dall'utente.
All'inizio di ogni episodio lo stato della cella viene inzializzato con un campione di alcune ore,
proveniente dal dataset a disposizione. 

Fino al termine dell'episodio di allenamento, l'agente ad ogni passo sceglie la temperatura a cui impostare
il glicole ed osserva la ricompensa che proviene dall'ambiente, salvando l'osservazione in un "replay buffer".
Prima di far avanzare la simulazione dell'episodio di un'unità di tempo, vengono aggiornati i pesi delle reti 
neurali che costituiscono l'agente (paragrafo successivo).

### Algoritmo scelto
Abbiamo scelto di implementare l'algoritmo DDPG ([link al paper](https://arxiv.org/abs/1509.02971)). L'agente
quindi è costituito da una coppia di reti neurali, `actorNN(s)`, e `criticNN(s, a)`. La prima sceglie che 
azione compiere dato lo stato corrente del sistema, mentre la seconda giudica il valore a lungo termine dell'
azione scelta.

### Shallow weight updates
Secondo comune pratica nell'ambito del Deep RL, le due reti che costituiscono l'agente sono affiancate altre
due reti equivalenti `target_actorNN(s)` e `target_criticNN(s, a)`, utilizzate nell'equazione di Bellman per 
migliorare le proprietà di convergenza e stabilità numerica del processo di allenamento. Si tratta della tecnica
"shallow weight update".