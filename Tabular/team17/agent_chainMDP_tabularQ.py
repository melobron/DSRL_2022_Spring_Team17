from chain_mdp import ChainMDP
import numpy as np

class agent():
    
    def __init__(self, env, gamma = 0.99):

        self.gamma = gamma
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        self.Q_table = np.zeros(shape=(num_states,num_actions))
        self.alpha = 0.8

    def train(self, episodes=1000, alpha=0.8):
        for epi in range(episodes):
            self.state = np.argmax(self.env.reset())
            done = False
            cum_reward = 0.0
            states = [self.state]

            for step in range(self.env.max_nsteps):
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
    
    def action(self, env, state):
        epsilon = 0.1
        state = state[-1]
        
        for x in range(state.size):
            if state[x] == 0:
                state = x-1
                break
            if x == 9:
                state = 9
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.Q_table[state])

        return action

    def update(self, s, action, ns, reward):
        state = s[-1]
        next_state = ns[-1]
        for x in range(state.size):
            if state[x] == 0:
                state = x-1
                break
            if x == 9:
                state = 9
        for x in range(next_state.size):
            if next_state[x] == 0:
                next_state = x-1
                break
            if x == 9:
                next_state = 9
        target = reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * target