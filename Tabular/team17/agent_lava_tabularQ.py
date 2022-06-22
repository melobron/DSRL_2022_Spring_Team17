import numpy as np
from lava_grid import ZigZag6x10

class agent():
    
    def __init__(self, env):
        # self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        self.gamma = 0.99
        self.state = env.reset()
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        self.Q_table = np.zeros(shape=(num_states,num_actions))
        
        return
    
    
    def update(self, s, action, ns, reward):
        next_state = np.argmax(ns)
        s = s[-1]
        state = np.argmax(s)
        target = reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action]
        self.Q_table[self.state, action] += 0.8 * target

    def action(self, env, state):
        state = state[-1]
        epsilon = 0.1
        state = np.argmax(state)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.Q_table[state])
        return action