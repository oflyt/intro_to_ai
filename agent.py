import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from collections import deque
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import itertools
import random
import h5py

class Agent:

    def __init__(self, state_size, action_size, stored_model="nothing"):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.memory = deque(maxlen=2000)
        #self.memory = np.zeros(2000, dtype=object)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005
        self.learning_rate = 0.01
        self.model = self._buildModel() if stored_model == "nothing" else load_model(stored_model)
        self.target_model = self._buildModel()
        self.tau = .05


    def _buildModel(self):
        model = Sequential()
        model.add(Conv2D(32,
                                8,
                                strides=(4, 4),
                                padding="valid",
                                activation="relu",
                                input_shape=self.state_size,
                                data_format="channels_first"))
        model.add(Conv2D(64,
                                4,
                                strides=(2, 2),
                                padding="valid",
                                activation="relu",
                                input_shape=self.state_size,
                                data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.action_size, activation="sigmoid"))
        model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1.0, clipvalue=0.5), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # def fitBatchFirst(self, batch_size=32):
    #     if len(self.memory) < batch_size:
    #         return
    #     samples = random.sample(self.memory, batch_size)
    #     states = []
    #     targets = []
    #     for sample in samples:
    #         state, action, reward, new_state, done = sample
    #         #target_val = max(self.model.predict(np.expand_dims(state, axis=0))[0])
    #         target = np.zeros(4)
    #         #target[action] = target_val
    #         print("first prediction")
    #         print(target[action])
    #
    #
    #         if done:
    #             target[action] = reward
    #         else:
    #             # future_q_array = self.model.predict(np.expand_dims(new_state, axis=0))[0]
    #             # print("Q-Array")
    #             # print(future_q_array)
    #             # Q_future = max([0,max(future_q_array)])
    #             # print("reward + (Q_future * self.gamma) - target[action]")
    #             # print(reward + (Q_future * self.gamma) - target[action])
    #             target[action] = target[action] + 0.1 * (reward + 0 - target[action])
    #             print("new target")
    #             print(target[action])
    #         states.append(state)
    #         targets.append(target)
    #     self.model.fit(np.array(states), np.array(targets), epochs=1, batch_size=batch_size, verbose=0)

    def fitBatch(self, batch_size=32):
        if len(self.memory) < batch_size: 
            return
        samples = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            target_val = max(self.target_model.predict(np.expand_dims(state, axis=0))[0])
            target = np.zeros(4)
            target[action] = target_val
            # print("first prediction")
            # print(target[action])

            if done:
                target[action] = reward
                future_q_array = self.target_model.predict(np.expand_dims(state, axis=0))[0]
                # print("Final-Prediction-Array")
                # print(future_q_array)
            else:
                future_q_array = self.target_model.predict(np.expand_dims(new_state, axis=0))[0]
                # print("Q-Array")
                # print(future_q_array)
                Q_future = max(future_q_array)
                # print("reward + (Q_future * self.gamma) - target[action]")
                # print(reward + (Q_future * self.gamma) - target[action])
                # print("target[action] + 0.1 * (reward + (Q_future * self.gamma) - target[action])")
                # print(target[action] + 0.1 * (reward + (Q_future * self.gamma) - target[action]))
                target[action] = target[action] + 0.1 * (reward + (Q_future * self.gamma) - target[action])
                #target[action] = reward + Q_future * self.gamma
                # print("new target")
                # print(target[action])
            states.append(state)
            targets.append(target)
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, batch_size=batch_size, verbose=0)
        print(history.history['loss'])

    def findAction(self, state):
        guess = self.model.predict(state)
        guess = np.argmax(guess)
        return guess

    def addToMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def saveToDisk(self, filename):
        self.model.save(filename)

    # def target_train(self):
    #     weights = self.model.get_weights()
    #     target_weights = self.target_model.get_weights()
    #     for i in range(len(target_weights)):
    #         target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    #     self.target_model.set_weights(target_weights)
