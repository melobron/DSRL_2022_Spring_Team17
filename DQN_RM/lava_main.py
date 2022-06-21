import argparse
import sys
from lava_agent import Agent

# Arguments
parser = argparse.ArgumentParser(description='Train DQN')

parser.add_argument("--gpu_num", type=int, default=0, help='gpu number')

# Basic Parameters
parser.add_argument("--state_size", type=int, default=10, help='size of states')
parser.add_argument("--action_size", type=int, default=2, help='size of actions')

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

args = parser.parse_args()

agent = Agent(args)
agent.train()
