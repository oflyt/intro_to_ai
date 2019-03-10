import random
import numpy as np
frames = 0
class Agent:
    def __init__(self, eps_start, eps_end, eps_steps, num_actions, gamma, gamma_n, n_step_return, brain):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.num_actions = num_actions
        self.gamma = gamma
        self.gamma_n = gamma_n
        self.n_step_return = n_step_return
        self.memory = []  # used for n_step return
        self.R = 0.
        self.brain = brain

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)

        else:
            s = np.array([s])
            p = self.brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(self.num_actions, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(self.num_actions)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * self.gamma_n) / self.gamma

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / self.gamma
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= self.n_step_return:
            s, a, r, s_ = get_sample(self.memory, self.n_step_return)
            self.brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect