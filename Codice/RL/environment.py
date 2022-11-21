import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import randint


class AppleStorageCell:

    # Stato della cella:
    #     matrice S[past_window][3], colonne:
    #     - temperatura cella
    #     - stato pompa
    #     - temperatura glicole linea mandata

    TEMP_IDX = 0
    PUMP_IDX = 1
    GLYCOL_IDX = 2

    def __init__(self, data_source, temp_model_on, temp_model_off, glycol_model,
                 reward_func_on, reward_func_off, past_window, time_unit, min_glycol_temp,
                 max_glycol_temp, temp_hysteresis, temp_setpoint, debug=False):

        # Dataset da cui vengono estratti gli stati iniziali degli episodi di simulazione
        # ( vedere AppleStorageCell.reset_state() )
        self.data_source = data_source

        # Funzioni che attribuiscono una ricompensa sulla base dello stato corrente della cella.
        # Una funzione per quando la pompa è attiva, ed una per quando è spenta
        self.reward_func_on = reward_func_on
        self.reward_func_off = reward_func_off

        # Numero di campioni che costituiscono lo stato corrente della cella
        self.past_window = past_window

        # Unità di tempo della simulazione, in minuti. Se per esempio il valore è pari a 5, la simulazione
        # si muove in avanti di 5 minuti alla volta
        self.time_unit = time_unit

        # Flag per visualizzare l'intera evoluzione dello stato della cella durante un episodio di simulazione
        self.debug = debug
        if self.debug:
            self.state_replay = None

        # le reti neurali che simulano l'ambiente della cella
        self.temp_model_on = temp_model_on
        self.temp_model_off = temp_model_off
        self.glycol_model = glycol_model

        # Temperatura massima e minima del glicole
        self.max_glycol_temp = max_glycol_temp
        self.min_glycol_temp = min_glycol_temp

        # Temperature di setpoint ed isteresi
        self.temp_hysteresis = temp_hysteresis
        self.temp_setpoint = temp_setpoint

        self.state = None

    # La funzione seguente ripristina lo stato della cella ad una finestra di <past_window> valori, campionati
    # da un dataset contenente valori reali

    def reset_state(self):
        start = randint(0, self.data_source.shape[0] - self.past_window)
        self.state = self.data_source.iloc[start:start + self.past_window].values
        if self.debug:
            self.state_replay = self.state.copy()

        self.state = np.reshape(self.state, (1, self.past_window, self.data_source.shape[1]))
        return self.state

    # La funzione accetta come argomento un valore <glycol_delta> che rappresenta la differenza
    # tra il valore attuale della temperatura del glicole sulla linea di mandata e la temperatura desiderata.
    # Tale valore è prodotto dall'agente che vogliamo allenare.
    # Il modo in cui viene aggiornato lo stato della cella, sulla base della temperatura del glicole è diverso
    # a seconda che la pompa, e quindi il sistema di raffreddamento, sia accesa oppure no.

    def update_state(self, glycol_delta):
        # Numero di iterazioni della simulazione che si verificano. Quando la pompa è spenta può essere
        # diverso da 1
        iterations = 1

        if self.pump_on():
            self.update_state_on(glycol_delta)
            reward = self.reward_func_on(self.current_glycol_temp())
        else:
            off_time, iterations = self.update_state_off()
            reward = self.reward_func_off(off_time)
        return self.state, reward, iterations

    # Funzione che aggiorna lo stato corrente della cella quando la pompa è accesa. Vengono calcolati i valori
    # della temp. del glicole nei <time_unit> minuti successivi, ovvero <temp_attuale + delta>, e sulla base di
    # questi viene calcolato il resto delle variabili di stato.

    def update_state_on(self, glycol_delta):
        # La temperatura del glicole sulla linea di mandata non può superare il valore sulla linea di ritorno, essendo
        # il risultato della miscelazione del glicole sulla linea di ritorno con un glicole più freddo
        future_glycol_temp = self.current_glycol_temp() + glycol_delta
        if future_glycol_temp < self.min_glycol_temp:
            future_glycol_temp = self.min_glycol_temp
        elif future_glycol_temp > self.max_glycol_temp:
            future_glycol_temp = self.max_glycol_temp

        future_glycol_temp_tensor = np.ones((1, self.time_unit, 1)) * future_glycol_temp
        future_cell_temps = self.predict_temp(future_glycol_temp_tensor)
        future_pump_states = np.around(self.predict_pump_states())
        state_update = np.concatenate((future_cell_temps, future_pump_states, future_glycol_temp_tensor), axis=2)

        if self.debug:
            self.state_replay = np.concatenate((self.state_replay, state_update[0]), axis=0)

        self.state = np.concatenate((self.state[:, self.time_unit:, :], state_update), axis=1)

    # Funzione che aggiorna lo stato corrente della cella quando la pompa è accesa. In questa situazione
    # la temperatura impostata sulla linea di mandata non deve essere sotto controllo dell'agente. Di conseguenza
    # la simulazione viene fatta avanzare finché la pompa non torna ad accendersi.
    #
    # Poichè abbiamo verificato che, quando la pompa è spenta, può accadere che la rete neurale
    # incaricata di predirne lo stato futuro (accesa, spenta) inpieghi molte iterazioni prima di farla
    # accendere, abbiamo impostato un numero massimo di iterazioni consecutive per cui la pompa può
    # rimanere spenta. Questo valore corrisponde a <MAX_OFF_INTERVAL>

    def update_state_off(self):
        count = 0
        off_time = 0
        while not self.pump_on():
            future_cell_temps = self.predict_temp()
            future_glycol_temps = self.predict_glycol_temp()
            future_pump_states = self.predict_pump_states()

            state_update = np.concatenate((future_cell_temps, future_pump_states, future_glycol_temps), axis=2)
            if self.debug:
                self.state_replay = np.concatenate((self.state_replay, state_update[0]), axis=0)

            self.state = np.concatenate((self.state[:, self.time_unit:, :], state_update), axis=1)

            off_time += self.time_unit
            count += 1

        return off_time, count

    # Ritorna <True> se la pompa è attualmente accessa

    def pump_on(self):
        return self.pump_col()[0][-1][0] == 1

    # Ritorna il valore più recente della temperatura del glicole sulla linea di mandata

    def current_glycol_temp(self):
        return self.glycol_col()[0][-1][0]

    # Ritorna il valore più recente della temperatura della cella

    def current_cell_temp(self):
        return self.temp_col()[0][-1][0]

    # Ritorna una predizione della temperatura dell'aria nella cella
    # per il prossimi <time_unit> minuti
    #
    # IMPORTANTE:
    #     La rete neurale utilizzata quando la pompa è spenta si aspetta in input una matrice le cui colonne
    #     sono, in ordine:
    #         - temperatura cella
    #         - stato pompa
    #         - temperatura glicole linea mandata
    #     La rete neurale utilizzata quando la pompa è accesa invece si aspetta in input sia una matrice analoga
    #     alla precedente, che un vettore di <time_unit> valori di temperatura del glicole

    def predict_temp(self, glycol_temps=None):

        model_input = tf.convert_to_tensor(self.state)
        model = self.temp_model_off
        if glycol_temps is not None:
            model_input = (model_input, tf.convert_to_tensor(glycol_temps))
            model = self.temp_model_on
        prediction = model.predict(model_input, verbose=0)

        # le predizioni della rete vengono mediate per ridurre il rumore
        return np.ones((1, self.time_unit, 1)) * prediction.mean()

    # Lo stato della pompa viene simulato supponendo di conoscere le temperature di setpoint ed isteresi.
    # Se la pompa era accesa e la cella supera la temp. di isteresi, la pompa si accende. Se la pompa
    # era accesa e la cella scende al di sotto della temp. di setpoint, la pompa si spegne.

    def predict_pump_states(self):
        vec_generator = np.zeros
        pump_is_on = self.pump_on()
        cell_temp = self.current_cell_temp()
        if pump_is_on and cell_temp > self.temp_setpoint or not pump_is_on and cell_temp > self.temp_hysteresis:
            vec_generator = np.ones
        return vec_generator((1, self.time_unit, 1))

    # Ritorna una predizione della temperatura del glicole sulla linea di mandata
    # per il prossimi <time_unit> minuti
    #
    # IMPORTANTE:
    #     La rete neurale si aspetta in input una matrice le cui colonne
    #     sono, in ordine:
    #         - temperatura glicole linea mandata
    #         - temperatura cella
    #         - stato pompa

    def predict_glycol_temp(self):
        model_input = tf.convert_to_tensor(np.concatenate((self.glycol_col(), self.temp_col(), self.pump_col()), axis=2))
        prediction = self.glycol_model.predict(model_input, verbose=0)

        # le predizioni della rete vengono mediate per ridurre il rumore
        return np.ones((1, self.time_unit, 1)) * prediction.mean()

    # Ritorna la colonna del tensore di stato corrispondente alla temperatura dell'
    # aria nella cella

    def temp_col(self):
        return self.state[:, :, self.TEMP_IDX:self.TEMP_IDX + 1]

    # Ritorna la colonna del tensore di stato corrispondente allo stato della pompa del glicole (accesa, spenta)

    def pump_col(self):
        return self.state[:, :, self.PUMP_IDX:self.PUMP_IDX + 1]

    # Ritorna la colonna del tensore di stato corrispondente alla temperatura del glicole
    # sulla linea di ritorno

    def glycol_col(self):
        return self.state[:, :, self.GLYCOL_IDX:self.GLYCOL_IDX + 1]

    # Funzione di utility per plottare lo stato della cella

    def plot_state(self, replay_window):
        if replay_window < 0:
            replay_window = self.state_replay.shape[0]

        plt.title("Cell and Glycol temperatures")
        plt.plot(self.state_replay[-replay_window:, self.TEMP_IDX:self.TEMP_IDX + 1], label="cell temp")
        plt.plot(self.state_replay[-replay_window:, self.GLYCOL_IDX:self.GLYCOL_IDX + 1], label="glycol temp")
        plt.legend()
        plt.show()
        plt.title("Pump state")
        plt.plot(self.state_replay[-replay_window:, self.PUMP_IDX:self.PUMP_IDX + 1], color="red")
        plt.show()
