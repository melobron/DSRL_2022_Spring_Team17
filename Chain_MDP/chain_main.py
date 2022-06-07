import argparse

from agent_chainMDP import DQNAgent


# Arguments
parser = argparse.ArgumentParser(description='DQN')

parser.add_argument("--gpu_num", type=int, default=0, help='gpu number')
parser.add_argument("--random_seed", type=int, default=100, help='pytorch random seed')

# Basic Parameters
parser.add_argument("--load_model", type=bool, default=False, help='Whether to load pretrained model')
parser.add_argument("--state_size", type=int, default=10, help='size of states')
parser.add_argument("--action_size", type=int, default=2, help='size of actions')

# Training Parameters
parser.add_argument("--batch_size", type=int, default=32, help='batch size')
parser.add_argument("--lr", type=float, default=0.0025, help='learning rate')
parser.add_argument("--n_episodes", type=int, default=1000, help='number of episodes')
parser.add_argument("--train_start", type=int, default=1000, help='num of memories to fill in')
parser.add_argument("--memory_size", type=int, default=20000, help='size of memory tree')

# Action Parameters
parser.add_argument("--epsilon", type=float, default=1.0, help='epsilon at start')
parser.add_argument("--epsilon_min", type=float, default=0.01, help='epsilon at end')
parser.add_argument("--explore_step", type=int, default=10000, help='epsilon decay')
parser.add_argument("--discount_factor", type=float, default=0.9, help='discount factor')

args = parser.parse_args()


agent = DQNAgent(args)
agent.train()
