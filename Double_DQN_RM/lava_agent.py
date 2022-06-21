import torch
import torch.nn as nn
import torch.optim as optim
import math
from itertools import count

from utils import *
from models import *
from lava_env import *

# Agent
class Agent:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # State, Action sizes
        self.state_size = args.state_size
        self.action_size = args.action_size

        # Environment
        self.env = ZigZag6x10()

        # Training Parameters
        self.n_episodes = args.n_episodes
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps_end = args.eps_end
        self.eps_start = args.eps_start
        self.eps_decay = args.eps_decay
        self.target_update = args.target_update

        # Prioritized Replay Memory
        self.memory_size = args.memory_size
        self.memory = ReplayMemory(self.memory_size)

        # Models
        self.policy_net = DQN(n_states=1, n_actions=self.action_size).to(self.device)
        self.target_net = DQN(n_states=1, n_actions=self.action_size).to(self.device)
        self.target_net.eval()

        # Initialize target network
        self.update_target_net()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Criterion
        self.criterion = nn.SmoothL1Loss()

        # AUC evaluation
        self.cum_rewards = []
        self.episodes = []
        self.auc = []

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.*self.env._steps/self.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                action = self.policy_net(state).unsqueeze(dim=0).max(1)[1].view(1, 1)
                return action
        else:
            action = torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.int)
            return action

    def optimize_model(self):
        # print('memory size: {}, batch size: {}'.format(len(self.memory), self.batch_size))
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Prediction
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Target
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        actions = self.policy_net(non_final_next_states).max(1)[1].reshape(-1, 1)
        q_values = self.target_net(non_final_next_states)
        selected_q_values = q_values.gather(dim=1, index=actions).squeeze()

        next_state_values[non_final_mask] = selected_q_values.detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        for i_episode in range(self.n_episodes):
            state = self.env.reset()
            cum_reward = 0.0
            for t in count():
                action = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action.item())
                cum_reward += reward
                reward = torch.tensor([reward], device=self.device)

                if done:
                    next_state = None

                self.memory.push(torch.tensor(state, device=self.device).unsqueeze(dim=0),
                                 action,
                                 torch.tensor(next_state, device=self.device).unsqueeze(dim=0) if next_state is not None else None,
                                 reward)

                state = next_state

                self.optimize_model()

                # print("step:{} | next state:{} reward:{} done:{}".format(t+1, next_state, reward, done))

                if done:
                    # print("episode:{} cum_reward:{}".format(i_episode+1, cum_reward))
                    self.cum_rewards.append(cum_reward)
                    break

            if i_episode % self.target_update == 0:
                self.update_target_net()

        torch.save(self.policy_net.state_dict(), './trained_models/lava_agent.pt')
        # print('AUC: {}'.format(sum(self.cum_rewards) / len(self.cum_rewards)))

    def load_weights(self):
        self.policy_net.load_state_dict(torch.load('./trained_models/lava_agent.pt'))

