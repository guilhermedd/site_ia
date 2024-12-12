import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, actor_critic, optimizer, gamma, player):
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state, legal_moves):
        """Seleciona uma ação com base na política atual, considerando apenas ações legais."""
        logits, _ = self.actor_critic(state[None, :])
        action_probs = tf.nn.softmax(logits).numpy()[0]

        if np.any(np.isnan(action_probs)) or np.any(action_probs < 0):
            action_probs = np.ones_like(action_probs) / len(action_probs)

        # Zerar as probabilidades de ações ilegais
        masked_probs = np.zeros_like(action_probs)
        masked_probs[legal_moves] = action_probs[legal_moves]

        if np.sum(masked_probs) == 0:
            # Caso todas as probabilidades legais sejam zero, distribui uniformemente entre as ações legais
            masked_probs[legal_moves] = 1.0 / len(legal_moves)
        else:
            # Re-normalizar as probabilidades
            masked_probs /= np.sum(masked_probs)

        # Selecionar uma ação com base nas probabilidades mascaradas
        return np.random.choice(len(masked_probs), p=masked_probs)

    def train_step(self, state, action, reward, next_state, done):
        """Realiza uma etapa de treinamento."""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits, value = self.actor_critic(state[None, :])
            _, next_value = self.actor_critic(next_state[None, :])

            action_probs = tf.nn.softmax(logits)
            log_prob = tf.math.log(action_probs[0, action])
            advantage = reward + self.gamma * next_value * (1 - int(done)) - value

            actor_loss = -log_prob * advantage
            critic_loss = advantage ** 2
            loss = actor_loss + critic_loss

        grads = tape.gradient(loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))
        return loss