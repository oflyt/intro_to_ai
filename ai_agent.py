from keras.layers.core import Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

import ai_util as util

class Agent:
    
    def __init__(self, state_shape, n_actions, epsilon=1.0):
        self.memory = deque(maxlen=500)
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

    def new_model(self):
        model = Sequential()
        model.add(Conv2D(16,
                        kernel_size=(4,4),
                        strides=(2, 2),
#                         padding="valid",
                        activation="relu",
                        input_shape=self.state_shape))
        model.add(Conv2D(32,
                        kernel_size=(4,4),
                        strides=(2, 2),
#                         padding="valid",
                        activation="relu",
                        input_shape=self.state_shape))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.n_actions))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        
    def save_model(self, name):
        try:
            self.model.save(name)
        except KeyboardInterrupt:
            self.model.save(name) 
            raise
        
    def load_model(self, name):
        self.model = load_model(name) 
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        state = np.moveaxis(state, 0, -1)
        Q_values = self.model.predict(np.stack([state]))[0]
        return Q_values.argmax(axis=0)
    
    def remember(self, state, action, reward, next_state, done):
        state = np.moveaxis(state, 0, -1)
        next_state = np.moveaxis(next_state, 0, -1)
        self.memory.append(np.array([state, action, reward, next_state, done]))
        
    def replay(self, batch_size):
        if len(self.memory) > batch_size:
            batch = np.array(random.sample(self.memory, batch_size))
            self._fit(self.model, batch, self.gamma, self.n_actions)
    
    def _fit(self, model, batch, gamma, n_outputs):
        states, actions, rewards, next_states, done = self._split_batch(batch, n_outputs)
        
        # Predict future
        predicted_future_Q_values = model.predict(next_states)
        predicted_future_rewards = predicted_future_Q_values.max(axis=1)
        
        # Calculate expected q values
        not_done_target = np.logical_not(done) * np.add(rewards, np.multiply(predicted_future_rewards, gamma))
        done_targets = done * rewards
        targets = np.add(not_done_target, done_targets)
        
        # Set expected q values for the actions in question
        target_Q_values = self.model.predict(states)
        target_Q_values[actions] = targets
        
        model.fit(states, target_Q_values, epochs=1, verbose=0)
            
    def _split_batch(self, batch, n_outputs):
        states, actions, rewards, next_states, done = np.array(np.split(batch, batch.shape[1], axis=1))[:, :, 0]
        actions = util.one_hot_encode(n_outputs, actions) 
        states = np.stack(states)
        next_states = np.stack(next_states)
        return (states, actions, rewards, next_states, done)

