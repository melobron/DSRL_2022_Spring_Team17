import numpy as np
import torch
import csv

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
            action = agent.action(env, s)
            ns, reward, done, _ = env.step(action)
            ns = np.reshape(ns, [1, state_size])
            s = ns
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)



def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []
    
    state_size = env.observation_space.n

    total_returns = []
    
    for epi in range(episodes):
        
        s = env.reset()
        s = np.reshape(s, [1, state_size])

        done = False
        cum_reward = 0.0

        rewards_in_epi = []

        agent.epi_update(epi)

        while not done:    
            action = agent.action(epi, s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            rewards_in_epi.append(reward)
            ns = np.reshape(ns, [1, state_size])
            
            agent.update(s, action, ns, reward, epi)

            s = ns
        
        episodic_returns.append(cum_reward)
        total_returns.append(rewards_in_epi)

    with open('./data/ChianMDP_PSRL.csv','w') as file :
        write = csv.writer(file)
        for episodic_return in episodic_returns:
            write.writerow([episodic_return])

    with open('./data/ChianMDP_PSRL_total_data.csv','w') as file :
        write = csv.writer(file)
        for rewards in total_returns:
            write.writerow(rewards)
                    
    return np.sum(episodic_returns)

