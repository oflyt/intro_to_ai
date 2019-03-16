import numpy as np
import random
# from collections import deque
from numpy_ringbuffer import RingBuffer
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.initializers import VarianceScaling
import tensorflow as tf
import keras
from custom_tensorboard import TensorBoardCustom

import ai_util as util


class Agent:
    
    def __init__(self, state_shape, n_actions, epsilon=1.0):
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9997

        self.capacity = 20000
        self.memory_states = RingBuffer(capacity=self.capacity, dtype=(np.uint8, state_shape))
        self.memory_next_states = RingBuffer(capacity=self.capacity, dtype=(np.uint8, state_shape))
        self.memory_action = RingBuffer(capacity=self.capacity, dtype=np.uint8)
        self.memory_reward = RingBuffer(capacity=self.capacity, dtype=np.uint16)
        self.memory_done = RingBuffer(capacity=self.capacity, dtype=np.bool)
        self.tensorboard = TensorBoardCustom('.log/run1/')

    def huber_loss(self, y_true, y_pred):
        return tf.losses.huber_loss(y_true, y_pred)

    def new_model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                         input_shape=self.state_shape,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2)))
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                         input_shape=self.state_shape,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2)))
        model.add(Flatten())
        model.add(Dense(256, activation="relu", kernel_initializer=tf.variance_scaling_initializer(scale=2)))
        model.add(Dense(self.n_actions, activation="linear", kernel_initializer=tf.variance_scaling_initializer(scale=2)))
        model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])
        self.model = model

    def save_model(self, name):
        try:
            self.model.save(name)
        except KeyboardInterrupt:
            self.model.save(name) 
            raise
        
    def load_model(self, name):
        self.model = load_model(name, custom_objects={'VarianceScaling': tf.variance_scaling_initializer(scale=2)}) #tf.variance_scaling_initializer(scale=2)})
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        state = np.moveaxis(state, 0, -1)
        Q_values = self.model.predict(np.stack([state]))[0]
        return Q_values.argmax(axis=0)
    
    def remember(self, state, action, reward, next_state, done):
        state = np.moveaxis(state, 0, -1)
        next_state = np.moveaxis(next_state, 0, -1)
        # self.memory.append(np.array([state, action, reward, next_state, done]))
        self.memory_action.append(action)
        self.memory_done.append(done)
        self.memory_next_states.append(next_state)
        self.memory_reward.append(reward)
        self.memory_states.append(state)

    def replay(self, batch_size, tot_reward):
        mem_size = len(self.memory_states)
        self.tensorboard.episode_score = tot_reward
        self.tensorboard.done = True
        if mem_size > batch_size:
            indices = np.random.choice(mem_size, batch_size, replace=False)

            states = self.memory_states[indices]
            next_states = self.memory_next_states[indices]
            actions = self.memory_action[indices]
            rewards = self.memory_reward[indices]
            done = self.memory_done[indices]

            self._fit(self.model, self.gamma, states, next_states, actions, rewards, done)
    
    def _fit(self, model, gamma, states, next_states, actions, rewards, done):
        # states, actions, rewards, next_states, done = self._split_batch(batch, n_outputs)
        
        # Predict future
        predicted_future_Q_values = model.predict(next_states)
        predicted_future_rewards = predicted_future_Q_values.max(axis=1)
        
        # Calculate expected q values
        not_done_target = np.logical_not(done) * np.add(rewards, np.multiply(predicted_future_rewards, gamma))
        done_targets = done * rewards
        targets = np.add(not_done_target, done_targets)
        
        # Set expected q values for the actions in question
        target_Q_values = self.model.predict(states)
        target_Q_values[range(len(actions)), actions] = targets

        model.fit(states, target_Q_values, epochs=1, verbose=0, callbacks=[self.tensorboard])

    # def _split_batch(self, batch, n_outputs):
    #     states, actions, rewards, next_states, done = np.array(np.split(batch, batch.shape[1], axis=1))[:, :, 0]
    #     actions = util.one_hot_encode(n_outputs, actions)
    #     states = np.stack(states)
    #     next_states = np.stack(next_states)
    #     return (states, actions, rewards, next_states, done)

