# Used to determine if explotation or exploration
import random

# image_processor.preprocess(param) returns
# an image that is grayscaled and half in size 
from image_preprocesser import ImagePreprocessor
image_processor = ImagePreprocessor()

# imports the gym module, which is the environment
# in which we will train our model
import gym
# we will be running our model on breakout
env = gym.make('BreakoutDeterministic-v4')  

# loads the agent that is to be used to play the game 
# params:
#   input_dimensions:   the dimensions of the input
#                       in breakout the dimensions are (4, 105, 80) - four images 105x80
#
#   output_dimensions:  the dimensions of the output
#                       in the gym environment you have access to the number of actions 
#                       through the env module

input_dimension     = (4, 105, 80)
output_dimension    = env.action_space.n

# The agent loads the weights of the previous model for continuous training
# unless there has been no model saved. The model name is by default model.h5
from agent import Agent
agent = Agent(input_dimension=input_dimension, output_dimension=output_dimension)


# First we will fill the memory of our agent. The items stored in the memory are 
# the following: 
#     previous_state, action, reward, current_state, done

# The memory of the states will look like the following:
# [
#     image_1,
#     image_2,
#     image_3,
#     image_4,
#     image_5,
#     image_6
# ]

# Where previous state will consist of image_1, image_2, image_3 and image_4.
# current_state will consist of image_3, image_4, image_5 and image_6

from memory import Memory
memory = Memory()

from file_handler import FileHandler
file_handler = FileHandler()
steps = file_handler.get_steps()


def run_test():
    print("\n\nTotally random, 10000 games")
    tot_reward = 0
    episode_range = 1000
    #Totally random gameplay
    for episode in range(episode_range):
        env.reset()
        done = False
 
        while not done:

            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            tot_reward += reward
        print(episode)
    print("Average score/game: {}".format(tot_reward / episode_range))


    agent.epsilon = 0.3
    tot_reward = 0
    for episode in range(episode_range):
    
        # Reseting variables for current game
        env.reset()
        done = False

        
        while not done:

            # Exploration/explotation determination
            # random.random() gives a value between 0 and 1
            if random.random() < agent.epsilon:
                action = env.action_space.sample()
            else:
                if len(memory.temp_memory) >= 6:
                    previous_state = memory.get_previous_state_from_tmp_memory()
                    action = agent.find_action(previous_state)
                else:
                    action = env.action_space.sample()
            # We will repeat the action 4 times
            # This is because we want to be able to track
            # the trajectory of the ball and so on
            for i in range(0, 4):
                state, reward, done, _ = env.step(action)
                tot_reward += reward
                state = image_processor.preprocess(state)
                memory.add_to_temp_memory(state)

            # If 6 or more images exist in the temporary memory, start
            # adding the states to the agents memory
            if len(memory.temp_memory) >= 6:
                previous_state = memory.get_previous_state_from_tmp_memory()
                current_state = memory.get_current_state_from_tmp_memory() 
        print(episode)
    print("Average score/game: {}".format(tot_reward / episode_range))


# run_test()

for episode in range(1000000):
    
    # Reseting variables for current game
    env.reset()
    done = False
    tot_reward = 0
    
    while not done:

        # Exploration/explotation determination
        # random.random() gives a value between 0 and 1
        if random.random() < agent.epsilon:
            action = env.action_space.sample()
        else:
            previous_state = memory.get_previous_state_from_tmp_memory()
            action = agent.find_action(previous_state)
        
        # We will repeat the action 4 times
        # This is because we want to be able to track
        # the trajectory of the ball and so on
        for i in range(0, 4):
            state, reward, done, _ = env.step(action)
            tot_reward += reward
            state = image_processor.preprocess(state)
            memory.add_to_temp_memory(state)

        # If 6 or more images exist in the temporary memory, start
        # adding the states to the agents memory
        if len(memory.temp_memory) >= 6:
            previous_state = memory.get_previous_state_from_tmp_memory()
            current_state = memory.get_current_state_from_tmp_memory() 
            memory.add_to_memory(previous_state, action, reward, current_state, done)


        # 20 000 states should have been recorded before the model 
        # should begin to train on the data (according to TA)
        if len(memory.memory) >= 20000:
            batch = memory.get_batch(batch_size=32)
            agent.fit(batch)

            agent.save_model()

        #env.render()
    if len(memory.memory) >= 20000:
        steps += 1  
        file_handler.write_to_file(steps)
        print("Total score in this game: {}".format(tot_reward))
    