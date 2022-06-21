import numpy as np
import torch

def calculate_performance(episodes, env, agent):

    episodic_returns = []
    
    state_size = env.observation_space.n    
    action_size = env.action_space.n
    
    for epi in range(episodes):
        
        s = env.reset()
        s = np.reshape(s, [1, state_size])

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            ns = np.reshape(ns, [1, state_size])
            s = ns
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)



def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []
    
    state_size = env.observation_space.n
    
    for epi in range(episodes):
        
        s = env.reset()
        s = np.reshape(s, [1, state_size])

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            ns = np.reshape(ns, [1, state_size])

            agent.append_sample(s, action, reward, ns, done)
            if agent.memory.tree.n_entries >= agent.train_start:
                agent.train_model()
            s = ns
        
        episodic_returns.append(cum_reward)

        torch.save(agent.model, "chain_agent")
                    
    return np.sum(episodic_returns)

