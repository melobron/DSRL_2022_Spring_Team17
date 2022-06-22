import numpy as np
import torch
import csv

def calculate_performance(episodes, env, agent):

    episodic_returns = []
    
    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            s = ns
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []

    state_size = env.observation_space.n
    action_size = env.action_space.n

    total_returns = []
    
    for epi in range(episodes):
        
        s = env.reset()

        if type(s) == np.int64:
            s = np.zeros((60,)).tolist()

        s = np.reshape(s, [1, state_size])

        done = False
        cum_reward = 0.0

        rewards_in_epi = []

        agent.epi_update(epi)

        while not done:    
            action = agent.action(epi, s)
            ns, reward, done, _ = env.step(action)
            ns = np.reshape(ns, [1, state_size])
            cum_reward += reward
            
            agent.update(s, action, ns, reward, epi, done)

            
            rewards_in_epi.append(reward)
            s = ns

        episodic_returns.append(cum_reward)
        total_returns.append(rewards_in_epi)

    with open('./data/lava_PSRL.csv','w') as file :
        write = csv.writer(file)
        for episodic_return in episodic_returns:
            write.writerow([episodic_return])

    with open('./data/lava_PSRL_total_data.csv','w') as file :
        write = csv.writer(file)
        for rewards in total_returns:
            write.writerow(rewards)

    data = agent.give_data()
    with open('./data/lava_data.csv','w') as file :
        for sets in data:
            write = csv.writer(file)
            write.writerow(sets)

    return np.sum(episodic_returns)

