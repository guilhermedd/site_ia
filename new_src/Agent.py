import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self, actor, critic, input_size, output_size, learning_rate=1e-3, gamma=0.99):
        self.actor = actor
        self.critic = critic
        self.optimizer = tf.keras.optmizer.Adam(learning_rate=learning_rate)
        self.gamma = gamma


    # Get the best move from the Actor
    def select_action(self, state, legal_moves):
        state = tf.convert_to_tensor(state)
        action_probs = self.actor(state).numpy()[0]

        if np.any(np.isnan(action_probs)) or np.any(action_probs < 0):
            action_probs = np.ones_like(action_probs) / len(action_probs)

        masked_probs = np.zeros_like(legal_moves)
        masked_probs[legal_moves] = action_probs[legal_moves]

        # if all probs are 0, then choose randomly
        if np.all(masked_probs) == 0:
            masked_probs[legal_moves] = 1.0 / len(legal_moves)
        else:
            masked_probs /= np.sum(masked_probs)

        return np.random.choice(len(masked_probs), p=masked_probs)


    # Get the state evaluation from the Critic
    def evaluate(self, state):
        state = tf.convert_to_tensor(state)
        value = self.critic(state)
        return value


    # Calculates the expected future reward
    def compute_returns(self, next_value, rewards, masks):
        returns = []
        R = next_value
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            R = reward + self.gamma * R * mask
            returns.append(R)
        return tf.convert_to_tensor(returns)


    # Performs a single training step
    def update(self, state, action_log_probs, rewards, masks, next_state):
        next_state = tf.convert_to_tensor(next_state)
        state = tf.convert_to_tensor(state)

        # Calculando o valor do estado futuro
        next_values = self.critic(next_state)
        returns = self.compute_returns(next_values, rewards, masks)

        # Calculando o valor do estado atual
        current_state = self.critic(state)

        # Calculando a vantagem
        advantages = returns - tf.squeeze(current_state)

        # Calculando as perdas
        actor_loss = -tf.reduce_mean(action_log_probs * advantages)
        critic_loss = tf.reduce_mean(tf.square(advantages))

        loss = actor_loss + critic_loss

        # Backpropagation
        with tf.GradientTape() as tape:
            total_loss = actor_loss + critic_loss

        gradients = tape.gradient(total_loss, self.actor.treinable_variables + self.critic.treinable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.treinable_variables + self.critic.treinable_variables))

        return loss