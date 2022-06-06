import numpy as np
from lava_grid import ZigZag6x10

class agent():
    
    def __init__(self, max_steps, act_fail_prob, goal, numpy_state, gamma = 0.99):
        # self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        self.gamma = gamma
        self.env = ZigZag6x10(max_steps=max_steps, act_fail_prob=act_fail_prob, goal=goal, numpy_state=numpy_state)
        self.state = self.env.reset()
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        self.Q_table = np.zeros(shape=(num_states,num_actions))
        self.Q_table = self.train()
        
        return
    
    def train(self, episodes=1000, alpha=0.8):
        for epi in range(episodes):
            self.state = self.env.reset()
            done = False
            cum_reward = 0.0
            states = [self.state]

            for step in range(self.env.max_steps):
                epsilon = 0.1
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q_table[self.state])
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.argmax(next_state)
                states.append(next_state)
                # print(next_state)
                cum_reward += reward
                target = reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[self.state, action]
                self.Q_table[self.state, action] += alpha * target
                self.state = next_state
                if done:
                    break
            
            if epi % 50 == 0:
                print(f"episode {epi}: reward {cum_reward}")
                print(states)
            
        return self.Q_table
    
    def action(self):
        
        # return self.sample_actions.pop(0)
        return np.argmax(self.Q_table[self.state])