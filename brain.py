import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras.initializers import Orthogonal
from keras import backend as K

# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, state_size, action_size, loss_entropy, loss_v, learning_rate, min_batch, gamma_n):
        self.GAMMA_N = gamma_n
        self.NONE_STATE = np.zeros(state_size)
        self.MIN_BATCH = min_batch
        self.LEARNING_RATE = learning_rate
        self.LOSS_V = loss_v
        self.LOSS_ENTROPY = loss_entropy
        self.state_size = state_size
        self.action_size = action_size
        self.session = tf.Session()
        self.rewards = []
        self.total_episodes = 0
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications
        self.writer = tf.summary.FileWriter(".log/run_2",
                                            self.session.graph)

    def _build_model(self):
        print((None,) + self.state_size)
        l_input = Input(batch_shape=(None,) + self.state_size)
        #normalize = BatchNormalization()(l_input)
        conv_1 = Conv2D(filters=32,
                         kernel_size=8,
                         strides=(4, 4),
                         padding="valid",
                         activation="relu",
                         kernel_initializer=Orthogonal(),
                         use_bias=False,
                         data_format="channels_first")(l_input)
        conv_2 = Conv2D(filters=64,
                        kernel_size=4,
                         strides=(2, 2),
                         padding="valid",
                         activation="relu",
                         kernel_initializer=Orthogonal(),
                         use_bias=False,
                         data_format="channels_first")(conv_1)
        conv_3 = Conv2D(filters=64,
                        kernel_size=1,
                        strides=(1, 1),
                        padding="valid",
                        activation="relu",
                        kernel_initializer=Orthogonal(),
                        use_bias=False,
                        data_format="channels_first")(conv_2)
        flatten = Flatten()(conv_3)
        dense_1 = Dense(512, activation='relu')(flatten)

        out_actions = Dense(self.action_size, activation='softmax')(dense_1)
        out_value = Dense(1, activation='linear')(dense_1)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None,) + self.state_size)
        a_t = tf.placeholder(tf.float32, shape=(None, self.action_size))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = self.LOSS_V * tf.square(advantage)  # minimize value error
        entropy = self.LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                                    keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < self.MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < self.MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.stack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.stack(s_)
        s_mask = np.vstack(s_mask)
        reward_summary = tf.Summary(value=[tf.Summary.Value(tag="average_reward",
                                                            simple_value=np.mean(self.rewards[-100:]))])
        self.writer.add_summary(reward_summary, self.total_episodes)

        if len(s) > 5 * self.MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + self.GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(self.NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def add_rewards(self, reward):
        self.rewards.append(reward)
        self.total_episodes += 1
