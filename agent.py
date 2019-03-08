from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import optimizers

class Agent:

    def __init__(self, input_dimension, output_dimension, model_name='model.h5'):
        self.input_dimension    = input_dimension
        self.output_dimension   = output_dimension
        self.model_name         = model_name
        self.epsilon            = 0.995
        self.epsilon_min        = 0.4
        self.epsilon_decay      = 0.995
        self.gamma              = 0.95
        self.model              = self._load_model()

    
    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(16,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=self.input_dimension,
            data_format="channels_first"))
        
        model.add(Conv2D(32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=self.input_dimension,
            data_format="channels_first"))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.output_dimension, activation="linear"))

        model.compile(loss="mse", optimizer='adam')
        return model

    
    def _load_model(self):
        try:
            model = load_model(self.model_name)
            print("\n\nModel {} was found and being loaded.\n\n".format(self.model_name))
            return model

        except OSError:
            print("\n\nWarning: no model was found, creating a new model\n\n")
            model = self._build_model()
            return model
        

    def save_model(self):
        self.model.save(self.model_name)

    
    def fit(self, batch):
        batch = batch
        for sample in batch:
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

        # epsilon decay, so that more explotation moves are made
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay            


    def find_action(self, state):
        guess = self.model.predict(state)
        guess = np.argmax(guess)
        return guess