import gym
from gym import spaces
import random
import numpy as np
import time
import sys


# from memory import Memory
from agent import Agent
from image_processor import ImageProcessor

# def getPreviousFourImages(images, counter):
#     arr = [
#         images[counter-4],
#         images[counter-3],
#         images[counter-2],
#         images[counter-1],
#     ]
#     arr = np.expand_dims(arr, axis=0)
#     return arr

#np.set_printoptions(threshold=sys.maxsize)
stored_model = "the_model.h5"
env = gym.make('BreakoutDeterministic-v4')
image_processor = ImageProcessor()

state = env.reset()

#####

start = time.time()
state = env.reset()
dims = (0,0)


if type(env.observation_space) is spaces.Box:
    (height, width, rgb) = env.observation_space.shape
    dims = image_processor.scaledDimensions(width, height, 0.25)
else:
    raise SystemExit



#openCV inverts order of dims
current_state = np.zeros((5, dims[1], dims[0]), dtype=int)
print(current_state.shape)
print(env.action_space)
print(env.action_space.n)
agent = Agent(current_state.shape, env.action_space.n)
counter = 0

while counter < 2000:
    done = False
    print("start only env")
    env.reset()
    while not done:
        #env.render()
        previous_state = current_state
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        img = image_processor.preprocess(state, dims)
        current_state = np.roll(current_state, 1, axis=0)
        current_state[0] = img
        agent.addToMemory(previous_state, action, reward, current_state, done)
        counter += 1
        # print(current_state[0][47], current_state[1][47], current_state[2][47], current_state[3][47])
        #
        # #image_processor.show(img)
        # print(len(state), len(state[0]))
        # print(reward)


    end = time.time()

    print("End of dummy")
    print(end - start)

#####

#4 images, 105x80 in size, output is the number of possible actions as we one-hot encode
#Shouldn't have to hardcode input_dim (4, 105, 80)

#agent.fitBatchFirst(2000)

for episode in range(1000):

    #reset env, get first four images
    state = env.reset()

    #retrieve first four images
    counter = 0
    tmp_images = []

    for i in range(0, 5):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        img = image_processor.preprocess(state, dims)
        current_state = np.roll(current_state, 1, axis=0)
        current_state[0] = img

    done = False
    tot_reward = 0
    start = time.time()
    state = env.reset()
    print("start real")
    while not done:

        previous_state = current_state
        e = random.random()
        if e < agent.epsilon:
            action = env.action_space.sample()
        else:
            action = agent.findAction(np.expand_dims(current_state, axis=0))

        state, reward, done, _ = env.step(action)
        img = image_processor.preprocess(state, dims)
        counter += 1

        current_state = np.roll(current_state, 1, axis=0)
        current_state[0] = img

        agent.addToMemory(previous_state, action, reward, current_state, done)
        tot_reward += reward
        if counter % 32 == 0:
            agent.fitBatch(32)
            #agent.target_train()
        #shorten training time, we can still see improvements
        #env.render()
    agent.epsilon = agent.epsilon_min + (1.0 - agent.epsilon_min) * np.exp(-agent.epsilon_decay * episode)
    end = time.time()
    print("epsilon")
    print(agent.epsilon)
    print()
    print("real end")
    print(end - start)


    #Print score
    print("episode: {}/2500, score: {}"
            .format(episode, tot_reward))

agent.saveToDisk(stored_model);




    




env.close()