from tensorflow.keras import layers, Sequential, Model

class Actor(Model):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.model = Sequential([
            layers.Input(shape=(input_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(output_size, activation='softmax'),
        ])

    def call(self, state):
        return self.model(state)