from itertools import count
import math

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from utils import *


class Agent:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Training Parameters
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.n_actions = args.n_actions
        self.n_episodes = args.n_episodes
        self.target_update = args.target_update
        self.gamma = args.gamma

        # Models
        self.policy_net = DQN(n_actions=self.n_actions).to(self.device)
        self.target_net = DQN(n_actions=self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Loss
        self.criterion = nn.SmoothL1Loss()

        # Optimizer
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)

        # Replay Memory
        self.memory = ReplayMemory(1000)
    
    def action(self, state, step):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step / self.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).unsqueeze(dim=0).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch.unsqueeze(dim=1)).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.unsqueeze(dim=1)).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, env):
        for episode in range(self.n_episodes):
            env.reset()
            state = env.state
            state = torch.tensor(state, dtype=torch.float).unsqueeze(dim=0).to(self.device)

            for t in count():
                print(t)
                action = self.action(state, t)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                if not done:
                    next_state = torch.tensor(env.state, dtype=torch.float).unsqueeze(dim=0).to(self.device)
                else:
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize_model()

                if done:
                    break

            if episode % self.target_update:
                self.target_net.load_state_dict(self.policy_net.state_dict())















