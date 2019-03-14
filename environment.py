from agent_a3c import Agent
import threading
import gym
import numpy as np
import time
from image_processor import ImageProcessor

class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, brain, eps_start, eps_end, eps_steps, thread_delay, env_name, gamma, gamma_n, n_step_return, render=False):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(env_name)
        self.brain = brain
        self.agent = Agent(eps_start, eps_end, eps_steps, self.env.action_space.n, gamma, gamma_n, n_step_return, self.brain)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.thread_delay = thread_delay
        self.dims = (84,84)
        self.current_state = np.zeros((4, self.dims[1], self.dims[0]), dtype=np.uint8)
        self.image_processor = ImageProcessor()

    def addToCurrentState(self, state):
        self.current_state = np.roll(self.current_state, 1, axis=0)
        self.current_state[0] = self.image_processor.preprocess(state, self.dims, False)

    def getCurrentState(self):
        return np.transpose(self.current_state, (1, 2, 0))

    def runEpisode(self):
        self.addToCurrentState(self.env.reset())
        for i in range(3):
            sta, _, _, _ = self.env.step(self.env.action_space.sample())
            self.addToCurrentState(sta)
        R = 0
        while True:
            time.sleep(self.thread_delay)  # yield
            current = self.getCurrentState()
            if self.render: self.env.render()
            s = current
            a = self.agent.act(current)
            sta, r, done, info = self.env.step(a)
            self.addToCurrentState(sta)
            s_ = self.getCurrentState()

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)
            R += r

            if done or self.stop_signal:
                self.brain.add_rewards(R)
                self.brain.total_episodes += 1
                break
        if R > 11:
            print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True