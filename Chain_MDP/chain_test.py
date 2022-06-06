import argparse
import random
import torch
from chain_mdp import ChainMDP
from agent_chainMDP import Agent


# Arguments
parser = argparse.ArgumentParser(description='DQN')

parser.add_argument("--gpu_num", type=int, default=0, help='gpu number')
parser.add_argument("--random_seed", type=int, default=100, help='pytorch random seed')

# Training Parameters
parser.add_argument("--batch_size", type=int, default=128, help='batch size')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
parser.add_argument("--n_episodes", type=int, default=10, help='number of episodes')
parser.add_argument("--target_update", type=int, default=10, help='episode size of target update')
parser.add_argument("--gamma", type=float, default=0.999, help='gamma factor for q learning')

# Action Parameters
parser.add_argument("--eps_start", type=int, default=0.9, help='epsilon at start')
parser.add_argument("--eps_end", type=int, default=0.05, help='epsilon at end')
parser.add_argument("--eps_decay", type=int, default=200, help='epsilon decay')
parser.add_argument("--n_actions", type=int, default=2, help='size of action space')

opt = parser.parse_args()

# recieve 1 at rightmost stae and recieve small reward at leftmost state
env = ChainMDP(10)
s = env.reset()

agent = Agent(opt)
agent.train(env)

done = False
cum_reward = 0.0
# always move right left: 0, right: 1
step = 0
state = torch.tensor(1, dtype=torch.float)
print('Evaluation start')
while not done:
    action = agent.action(state, step)
    action = action.cpu().item()
    print(action)
    ns, reward, done, _ = env.step(action)
    print(ns, reward, done)
    cum_reward += reward
    step += 1
print(f"total reward: {cum_reward}")
