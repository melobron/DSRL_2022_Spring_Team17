import numpy as np
import torch


def calculate_performance(episodes, env, agent):
    episodic_returns = []

    for epi in range(episodes):

        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:
            action = agent.select_action(s)
            ns, reward, done, _ = env.step(action.item())
            cum_reward += reward
            s = ns
        # print('episode:{}, cum reward:{}'.format(epi, cum_reward))
        episodic_returns.append(cum_reward)

    # return np.sum(episodic_returns)
    return episodic_returns


def calculate_sample_efficiency(episodes, agent):
    agent.n_episodes = episodes
    agent.train()
    episodic_returns = agent.cum_rewards
    # return np.sum(episodic_returns)
    return episodic_returns
