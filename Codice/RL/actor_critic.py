import numpy as np
import tensorflow as tf


# Qui sono definite le architetture della rete 'actor', che sceglie l'azione da compiere,
# e la rete 'critic', che giudica la qualità dell'azione scelta

def get_actor(past_window, num_states, lower_bound, upper_bound):
    assert lower_bound < upper_bound

    inputs = tf.keras.layers.Input(shape=(past_window, num_states))
    lstm = tf.keras.layers.GRU(16, return_sequences=True, dropout=0.1,
                               recurrent_dropout=0.3)(inputs)
    flatten = tf.keras.layers.Flatten()(lstm)
    x = tf.keras.layers.Dense(16, activation='relu')(flatten)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    output = output*(upper_bound - lower_bound) + lower_bound

    return tf.keras.models.Model(inputs=inputs, outputs=output)


def get_critic(past_window, num_states, num_actions):
    state_inputs = tf.keras.layers.Input(shape=(past_window, num_states))
    encoder, state_h = tf.keras.layers.GRU(16, return_state=True,
                                           dropout=0.1, recurrent_dropout=0.3)(state_inputs)
    actions = tf.keras.Input(shape=(num_actions, 1))
    decoder = tf.keras.layers.GRU(16, return_sequences=True, dropout=0.1,
                                  recurrent_dropout=0.3)(actions, initial_state=state_h)

    x = tf.keras.layers.Dense(16, activation='relu')(decoder)
    output = tf.keras.layers.Dense(1, activation='linear')(x)

    return tf.keras.models.Model(inputs=[state_inputs, actions], outputs=output)


# Questa funzione gestisce l'allenamento delle reti neurali.
def learn(buffer, batch_size, target_actor, target_critic,
            critic_model, actor_model, critic_optimizer, 
            actor_optimizer, gamma):
  
    # Vengono scelti casualmente <batch_size> indici
    record_range = buffer.capacity if buffer.full else buffer.counter
    batch_indices = np.random.choice(record_range, batch_size)

    state_batch = tf.convert_to_tensor(buffer.state_buffer[batch_indices])
    action_batch = tf.convert_to_tensor(buffer.action_buffer[batch_indices])
    reward_batch = tf.convert_to_tensor(buffer.reward_buffer[batch_indices])
    reward_batch = tf.cast(reward_batch, dtype=tf.float32)
    next_state_batch = tf.convert_to_tensor(buffer.next_state_buffer[batch_indices])

    update(target_actor, target_critic, critic_model, actor_model, critic_optimizer, actor_optimizer, gamma, state_batch, action_batch, reward_batch, next_state_batch)


# Il decoratore segnala a tensorflow di compilare questa funzione in un grafo di calcolo,
# che utilizza le primitive di TF e risulta in un'esecuzione molto più veloce
@tf.function
def update(target_actor, target_critic, critic_model, actor_model,
           critic_optimizer, actor_optimizer, gamma, state_batch, 
          action_batch, reward_batch, next_state_batch):
  
    with tf.GradientTape() as tape:
        future_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic(
              [next_state_batch, future_actions], training=True
            )
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor_model(state_batch, training=True)
        critic_value = critic_model([state_batch, actions], training=True)
        # Massimizzare il valore assegnato all'azione scelta equivale a
        # minimizzare `-value`
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.trainable_variables)
    )

# Questa funzione regola l'update delle reti 'target', utilizzate
# per calcolare e giudicare le ipotetiche azioni future dell'agente.
# I pesi di queste reti vengono aggiornati più lentamente di quelli
# delle reti principali, per garantire stabilità.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
