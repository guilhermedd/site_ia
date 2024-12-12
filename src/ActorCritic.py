import tensorflow as tf
from tensorflow.keras import layers

class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int, **kwargs):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.num_hidden_units = num_hidden_units
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor):
        x = self.common(inputs)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_config(self):
        # Retorna os parâmetros necessários para recriar a instância
        config = super().get_config()
        config.update({
            "num_actions": self.num_actions,
            "num_hidden_units": self.num_hidden_units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Recria a instância com base na configuração
        return cls(**config)
