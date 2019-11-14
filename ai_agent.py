from keras.layers.core import Flatten
from keras.layers import Dense, Conv2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import numpy as np
import random
from numpy_ringbuffer import RingBuffer

class Agent:
    
    def __init__(self, state_shape, n_actions, epsilon=None):
        self.capacity = 100000
        self.state_memory = RingBuffer(capacity=self.capacity, dtype=(np.uint8, state_shape))
        self.next_state_memory = RingBuffer(capacity=self.capacity, dtype=(np.uint8, state_shape))
        self.action_memory = RingBuffer(capacity=self.capacity, dtype=np.uint8)
        self.reward_memory = RingBuffer(capacity=self.capacity, dtype=np.uint16)
        self.done_memory = RingBuffer(capacity=self.capacity, dtype=np.bool)
        
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.gamma = 0.99
        
        if epsilon is None:
            self.epsilon = 1.0
            self.epsilon_decay = 0.9997
            self.epsilon_min = 0.05
            self.decay_epsilon = True
        else:
            self.epsilon = epsilon
            self.decay_epsilon = False
        
        
        self.q = None

    def new_model(self):
        model = Sequential()

        model.add(Conv2D(16,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=self.state_shape))
        
        model.add(Conv2D(32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation="relu",
            input_shape=self.state_shape))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.n_actions, activation="linear"))

        model.compile(loss="mse", optimizer=Adam())#lr=0.1))

        self.model = model
        
    def save_model(self, name):
        try:
            self.model.save(name)
        except KeyboardInterrupt:
            self.model.save(name) 
            raise
        
    def load_model(self, name):
        self.model = load_model(name) 
        #self.model.optimizer.lr.assign(0.001)
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        
        state = np.moveaxis(state, 0, -1)
        Q_values = self.model.predict(np.stack([state]))[0]
        return Q_values.argmax(axis=0)
    
    def remember(self, state, action, reward, next_state, done):
        state = np.moveaxis(state, 0, -1)
        next_state = np.moveaxis(next_state, 0, -1)
        
        self.state_memory.append(state)
        self.next_state_memory.append(next_state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.done_memory.append(done)
        
    def replay(self, batch_size):
        memory_size = len(self.state_memory)
        if memory_size > batch_size:
            indecies = np.random.choice(memory_size, batch_size, replace=False)
            
            states = self.state_memory[indecies]
            next_states = self.next_state_memory[indecies]
            actions = self.action_memory[indecies]
            rewards = self.reward_memory[indecies]
            done = self.done_memory[indecies]
            
            self._fit(self.model, self.gamma, states, next_states, actions, rewards, done)
            
            if self.decay_epsilon == True:
                if self.epsilon <= self.epsilon_min:
                    self.epsilon = 1.0
                self.epsilon *= self.epsilon_decay
    
    def _fit(self, model, gamma, states, next_states, actions, rewards, done):
        # Predict future
        predicted_future_Q_values = model.predict(next_states)
        predicted_future_rewards = predicted_future_Q_values.max(axis=1)
        
        # Calculate expected q values
        not_done_targets = np.logical_not(done) * np.add(rewards, np.multiply(predicted_future_rewards, gamma))
        done_targets = done * rewards
        targets = np.add(not_done_targets, done_targets)
        
        # Set expected q values for the actions in question
        target_Q_values = model.predict(states)
        self.q = target_Q_values
        target_Q_values[range(len(actions)), actions] = targets
        
        model.fit(states, target_Q_values, epochs=1, verbose=0)
            
