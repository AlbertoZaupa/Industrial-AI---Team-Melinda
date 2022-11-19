import tensorflow as tf
from buffer import *


class Agent:

    def __init__(self, past_window_size, num_states, lower_bound, upper_bound, buffer_size, time_unit,
                 gamma, tau, lr):
        assert upper_bound > lower_bound

        # La rete neurale che sceglie le azioni da compiere, ovvero a che valore impostare la temp. del glicole
        self.actor = actor_nn(past_window=past_window_size, num_states=num_states,
                              lower_bound=lower_bound, upper_bound=upper_bound)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr)

        # La rete neurale che giudica la `qualità` delle azioni scelte, in base allo stato di arrivo che segue
        # l'azione scelta, e in base al valore degli stati raggiungibili dallo stato di arrivo
        self.critic_optimizer = tf.keras.optimizers.Adam(lr)
        self.critic = critic_nn(past_window=past_window_size, num_states=num_states)

        # Replay buffer utilizzato durante la fase di allenamento
        self.replay_buffer = ReplayBuffer(past_window=past_window_size, num_states=num_states,
                                          time_unit=time_unit, capacity=buffer_size)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.gamma = gamma
        self.tau = tau

    # Utilizzata durante la fase di allenamento
    def training_policy(self, state):
        action = self.actor.predict(state, verbose=0)
        # Viene aggiunto del rumore bianco alla temperatura scelta, in modo tale da permettere all'
        # agente di esplorare nuove traiettorie di azioni
        mean = 0
        stddev = 1
        noise = np.random.normal(mean, stddev, size=1)
        action = action + noise

        # la temperatura scelta deve sempre stare tra <lower_bound> e <upper_bound>
        clip = np.clip(action, self.lower_bound, self.upper_bound)
        return clip

    # Utilizzata nella fase di esplorazione iniziale
    def random_policy(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, size=1)

    def learn(self, target_actor, target_critic, batch_size):
        # Vengono scelti casualmente <BATCH_SIZE> indici
        record_range = self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.counter
        batch_indices = np.random.choice(record_range, batch_size)

        state_batch = tf.convert_to_tensor(self.replay_buffer.state_buffer[batch_indices])
        state_batch = tf.cast(state_batch, tf.float32)
        action_batch = tf.convert_to_tensor(self.replay_buffer.action_buffer[batch_indices])
        action_batch = tf.cast(action_batch, tf.float32)
        reward_batch = tf.convert_to_tensor(self.replay_buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, tf.float32)
        next_state_batch = tf.convert_to_tensor(self.replay_buffer.next_state_buffer[batch_indices])
        next_state_batch = tf.cast(next_state_batch, tf.float32)

        # vengono eseguita un'iterazione di gradient descent sulle reti che costituiscono l'agente
        # (critic_model, actor_model)
        self.SGD(target_actor=target_actor, target_critic=target_critic, state_batch=state_batch,
                 action_batch=action_batch, reward_batch=reward_batch, next_state_batch=next_state_batch)

    @tf.function
    def SGD(self, target_actor, target_critic, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            # Le reti `target` sono utilizzate per valutare le azioni eseguite negli stati di arrivo,
            # e la relativa qualitò. Non vengono riutilizzate le reti corrispondenti all'agente
            # migliorare la probabilità di convergenza del processo di allenamento

            future_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * target_critic(
                [next_state_batch, future_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(critic_value - y))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Massimizzare il valore assegnato all'azione scelta equivale a
            # minimizzare `-value`
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        # Vengono aggiornati i pesi delle reti `target` (target_actor, target_critic)
        update_target(target_actor.variables, self.actor.variables, self.tau)
        update_target(target_critic.variables, self.critic.variables, self.tau)


# Qui sono definite le architetture della rete 'actor', che sceglie l'azione da compiere,
# e la rete 'critic', che giudica la qualità dell'azione scelta

def actor_nn(past_window, num_states, lower_bound, upper_bound):
    assert lower_bound < upper_bound

    inputs = tf.keras.layers.Input(shape=(past_window, num_states))
    gru = tf.keras.layers.GRU(16, dropout=0.1, recurrent_dropout=0.3)(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(gru)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    output = output * (upper_bound - lower_bound) + lower_bound

    return tf.keras.models.Model(inputs=inputs, outputs=output)


def critic_nn(past_window, num_states):
    state_inputs = tf.keras.layers.Input(shape=(past_window, num_states))
    encoder, state_h = tf.keras.layers.GRU(16, return_state=True,
                                           dropout=0.1, recurrent_dropout=0.3)(state_inputs)
    actions = tf.keras.Input(shape=(1, 1))
    decoder = tf.keras.layers.GRU(16, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(actions, initial_state=state_h)

    x = tf.keras.layers.Dense(16, activation='relu')(decoder)
    output = tf.keras.layers.Dense(1, activation='linear')(x)

    return tf.keras.models.Model(inputs=[state_inputs, actions], outputs=output)


# Questa funzione regola l'update delle reti 'target', utilizzate
# per calcolare e giudicare le ipotetiche azioni future dell'agente.
# I pesi di queste reti vengono aggiornati più lentamente di quelli
# delle reti principali, per assicurare maggiore stabilità numerica all'allenamento
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


# Classe che rappresenta un agente già allenato, che è costituito solamente da
# una rete `actor`

class TrainedAgent:

    def __init__(self, actor_net):
        self.actor = actor_net

    def policy(self, state):
        return self.actor.predict(state, verbose=0)
