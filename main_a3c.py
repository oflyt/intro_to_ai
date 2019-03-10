# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K
from brain import Brain
from optimiser import Optimizer
from environment import Environment

# -- constants
ENV = 'BreakoutDeterministic-v4'

RUN_TIME = 100000
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient


# ---------
frames = 0

# -- main
dims = (84,84)

brain = Brain(state_size=(4,84,84), action_size=4, loss_entropy=LOSS_ENTROPY,
              loss_v=LOSS_ENTROPY, learning_rate=LEARNING_RATE, min_batch=MIN_BATCH,
              gamma_n=GAMMA_N)  # brain is global in A3C

envs = [Environment(brain=brain, render=False, eps_start=0.,
                    eps_end=0., eps_steps=EPS_STEPS, thread_delay=THREAD_DELAY,
                    env_name=ENV, gamma=GAMMA, gamma_n=GAMMA_N,
                    n_step_return=N_STEP_RETURN) for i in range(THREADS)]

opts = [Optimizer(brain) for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
env_test = Environment(brain=brain, render=True, eps_start=0., eps_end=0., eps_steps=EPS_STEPS, thread_delay=THREAD_DELAY, env_name=ENV, gamma=GAMMA, gamma_n=GAMMA_N, n_step_return=N_STEP_RETURN)
env_test.run()