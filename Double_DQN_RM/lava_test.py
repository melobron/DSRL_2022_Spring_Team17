import random
import numpy as np
import torch
import argparse

import os
import sys
import csv

from lava_env import *
from lava_agent import *
from lava_interaction import *


def evaluate_performance(args, seeds=(1, 2, 3, 4, 5)):
    episodes = 50
    pf_list = []

    for seed in seeds:
        print(f'Seed {seed} start...')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env_kwargs = {"max_steps":100, "act_fail_prob":0, "goal":(5, 9), "numpy_state":True}
        env = ZigZag6x10(**env_kwargs)

        agent = Agent(args)
        agent.env = env
        agent.load_weights()

        pf = calculate_performance(episodes, env, agent)
        pf_list.append(np.sum(pf))

        # Save CSV file
        f = open('./results/seed{}_Double_DQN_Lava_performance.csv'.format(seed), 'w', newline='')
        writer = csv.writer(f)
        for item in pf:
            writer.writerow([item])
        f.close()

    print(f'Avg Performance: {np.mean(pf_list)}')

    with open('./results/lava_pf.txt', 'a') as f:
        for seed in seeds:
            f.write('shape:{} | seed:{} | {}\n'.format(env.shape, seed, np.mean(pf_list)))


def evaluate_sample_efficiency(args, seeds=(1, 2, 3, 4, 5)):
    episodes = 3000
    se_list = []

    for seed in seeds:
        print(f'Seed {seed} start...')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env_kwargs = {"max_steps":100, "act_fail_prob":0, "goal":(5, 9), "numpy_state":True}
        env = ZigZag6x10(**env_kwargs)
        agent = Agent(args)
        agent.env = env

        se = calculate_sample_efficiency(episodes, agent)
        se_list.append(np.sum(se))

        # Save CSV file
        f = open('./results/seed{}_Double_DQN_Lava_sample_efficiency.csv'.format(seed), 'w', newline='')
        writer = csv.writer(f)
        for item in se:
            writer.writerow([item])
        f.close()

    print(f'Avg sample efficiency score : {np.mean(se_list)}')

    with open('./results/lava_se.txt', 'a') as f:
        for seed in seeds:
            f.write('shape:{} | seed:{} | {}\n'.format(env.shape, seed, np.mean(se_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN')

    parser.add_argument("--gpu_num", type=int, default=0, help='gpu number')

    # Basic Parameters
    parser.add_argument("--state_size", type=int, default=60, help='size of states')
    parser.add_argument("--action_size", type=int, default=4, help='size of actions')

    # Training Parameters
    parser.add_argument("--n_episodes", type=int, default=500, help='number of episodes for training')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--gamma", type=float, default=0.999, help='gamma')
    parser.add_argument("--eps_end", type=float, default=0.05, help='epsilon at end')
    parser.add_argument("--eps_start", type=float, default=0.9, help='epsilon at start')
    parser.add_argument("--eps_decay", type=float, default=200, help='epsilon decay')
    parser.add_argument("--target_update", type=int, default=10, help='update period')
    parser.add_argument("--memory_size", type=int, default=10000, help='size of replay memory')

    # Test Parameters
    parser.add_argument("--evalType", type=int, default=1, help='0: performance, 1: sample efficiency, 2: adaptability')
    args = parser.parse_args()

    seeds = [5]
    # seeds = (1, 2, 3, 4, 5)
    # seeds = (6, 7, 8, 9, 10)

    if args.evalType == 0:
        evaluate_performance(args, seeds)
    elif args.evalType == 1:
        evaluate_sample_efficiency(args, seeds)
