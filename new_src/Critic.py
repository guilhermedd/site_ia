from tensorflow.keras import Model, layers, Sequential

class Critic(Model):
    def __init__(self, input_size, **kwargs):
        super().__init__(**kwargs)
        self.model = Sequential([
            layers.Input(input_size=(input_size,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1)
        ])

    def call(self, state):
        return self.model(state)