import gym
from gym import spaces
import random
import numpy as np
import time



# from memory import Memory
from agent import Agent
from image_processor import ImageProcessor


#np.set_printoptions(threshold=sys.maxsize)
stored_model = "the_model_ddqn_5.h5"
env = gym.make('BreakoutDeterministic-v4')
image_processor = ImageProcessor()
replay_length = 200000
dims = (84,84)
current_state = np.zeros((4, dims[1], dims[0]), dtype=np.uint8)
agent = Agent(current_state.shape, env.action_space.n, replay_length)


#####

# if type(env.observation_space) is spaces.Box:
#     (height, width, rgb) = env.observation_space.shape
#     # dims = image_processor.scaledDimensions(width, height, 0.25)
# else:
#     raise SystemExit


print(env.unwrapped.get_action_meanings())
#openCV inverts order of dims

print(current_state.shape)
print(env.action_space)
print(env.action_space.n)

####PREFILL


state = env.reset()
start = time.time()


try:
    agent.memory_current = np.load('numpy_prefill_memory_current.npy')
    agent.memory_reward = np.load('numpy_prefill_memory_reward.npy')
    agent.memory_action = np.load('numpy_prefill_memory_action.npy')
    agent.memory_done = np.load('numpy_prefill_memory_done.npy')
    agent.memory_next = np.load('numpy_prefill_memory_next.npy')
except IOError:
    prefill_counter = 0
    print("start prefill")
    while prefill_counter < replay_length:
        done = False
        env.reset()
        print("counter = ", prefill_counter)
        while not done:
            previous_state = current_state
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            img = image_processor.preprocess(state, dims)
            current_state = np.roll(current_state, 1, axis=0)
            current_state[0] = img
            agent.addToMemory(previous_state, action, reward, current_state, done, prefill_counter)
            prefill_counter += 1

        end = time.time()

    print("End of prefill")
    print(end - start)
    np.save('numpy_prefill_memory_current', agent.memory_current)
    np.save('numpy_prefill_memory_reward', agent.memory_reward)
    np.save('numpy_prefill_memory_action', agent.memory_action)
    np.save('numpy_prefill_memory_done', agent.memory_done)
    np.save('numpy_prefill_memory_next', agent.memory_next)

#####

big_counter = 0

for episode in range(10000):

    #reset env, get first four images
    state = env.reset()

    #retrieve first four images
    counter = 0
    no_reward_counter = 0
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
    #lives = env.unwrapped.ale.lives()
    while not done and no_reward_counter < 400:

        previous_state = current_state
        e = random.random()
        if e < agent.epsilon:
            action = env.action_space.sample()
        else:
            action = agent.findAction(np.expand_dims(current_state, axis=0))

        state, reward, done, _ = env.step(action)
        img = image_processor.preprocess(state, dims)
        counter += 1
        big_counter += 1
        if reward == 0:
            no_reward_counter += 1
        else:
            no_reward_counter = 0

        current_state = np.roll(current_state, 1, axis=0)
        current_state[0] = img

        #lost_life_or_done = env.unwrapped.ale.lives() < lives or done

        agent.addToMemory(previous_state, action, reward, current_state, done, big_counter)
        tot_reward += reward
        agent.fitBatch(32)
        if (big_counter + 1) % 10000 == 0:
            print("train target", big_counter)
            agent.target_train()

        #env.render()
    agent.epsilon = agent.epsilon_min + (1.0 - agent.epsilon_min) * np.exp(-agent.epsilon_decay * (episode - 4000 if episode > 6000 else episode))
    #agent.target_train()
    end = time.time()
    print()
    print("decision vector", agent.getPredictionVector())
    print("frames played", counter)
    print("total frames played", big_counter)
    print("finished cleanly", done if done else no_reward_counter)
    print("epsilon: ", agent.epsilon)
    print("real end time: ", end - start)
    print("average per step", (end - start)/counter)
    # print("image process time per step", image_processor.processTime)
    # print("addToMemory time per step", agent.addToMemoryTime)
    # print("fitBatch time per step", agent.fitBatchTime)
    # print("findAction time per step", agent.findActionTime)
    agent.saveToDisk(stored_model, episode)


    #Print score
    print("episode: {}/10000, score: {}"
            .format(episode, tot_reward))

agent.saveToDisk(stored_model)
env.close()
