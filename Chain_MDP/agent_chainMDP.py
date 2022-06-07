import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from model import DQN
from chain_mdp import ChainMDP


class DQNAgent:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Load Model
        self.load_model = args.load_model

        # State, Action sizes
        self.state_size = args.state_size
        self.action_size = args.action_size

        # Environment
        self.env = ChainMDP(self.state_size)

        # Training Parameters
        self.n_episodes = args.n_episodes
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.explore_step = args.explore_step
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.discount_factor = args.discount_factor
        self.train_start = args.train_start

        # Prioritized Replay Memory
        self.memory_size = args.memory_size
        self.memory = Memory(self.memory_size)

        # Models
        self.policy_net = DQN(n_actions=self.action_size)#.to(self.device)
        self.target_net = DQN(n_actions=self.action_size)#.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Initialize target network
        self.update_target_net()

        # Load Model
        if self.load_model:
            self.policy_net = torch.load('chain_agent')

        # AUC evaluation
        self.cum_rewards = []
        self.episodes = []
        self.auc = []

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state, eval_mode=False):
        # Greedy for evaluation
        if eval_mode:
            state = torch.from_numpy(state)
            state = state.float().cpu()
            q_value = self.policy_net(state)
            _, action = torch.max(q_value, 1)
            return int(action)

        # Epsilon-greedy for training
        elif np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = state.float().cpu()
            q_value = self.policy_net(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    def append_sample(self, state, action, reward, next_state, done):
        target = self.policy_net(torch.FloatTensor(state)).data
        old_val = target[0][action]
        target_val = self.target_net(torch.FloatTensor(next_state)).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))

    def train(self):
        for episode in range(self.n_episodes):
            state = self.env.reset()
            state = state[np.newaxis, :]

            done = False
            cum_reward = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state[np.newaxis, :]
                self.append_sample(state, action, reward, next_state, done)

                if self.memory.tree.n_entries >= self.train_start:
                    if self.epsilon > self.epsilon_min:
                        self.epsilon -= self.epsilon_decay

                    mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
                    mini_batch = np.array(mini_batch).transpose()

                    states = np.vstack(mini_batch[0])
                    actions = list(mini_batch[1])
                    rewards = list(mini_batch[2])
                    next_states = np.vstack(mini_batch[3])
                    dones = mini_batch[4]

                    # bool to binary
                    dones = dones.astype(int)

                    # Q function of current state
                    states = torch.Tensor(states)
                    states = states.float()
                    pred = self.policy_net(states)

                    # one-hot encoding
                    a = torch.LongTensor(actions).view(-1, 1)

                    one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
                    one_hot_action.scatter_(1, a, 1)

                    pred = torch.sum(pred.mul(one_hot_action), dim=1)

                    # Q function of next state
                    next_states = torch.Tensor(next_states)
                    next_states = next_states.float()
                    next_pred = self.target_net(next_states).data

                    rewards = torch.FloatTensor(rewards)
                    dones = torch.FloatTensor(dones)

                    # Q Learning: get maximum Q value at s' from target model
                    target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]

                    errors = torch.abs(pred - target).data.numpy()

                    # update priority
                    for i in range(self.batch_size):
                        idx = idxs[i]
                        self.memory.update(idx, errors[i])

                    self.optimizer.zero_grad()

                    # MSE Loss function
                    loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
                    loss.backward()

                    # and train
                    self.optimizer.step()

                cum_reward += reward
                state = next_state

                if done:
                    self.update_target_net()

                    self.cum_rewards.append(cum_reward)
                    self.episodes.append(episode)
                    if not self.auc:
                        self.auc.append(cum_reward)
                    else:
                        self.auc.append(self.auc[-1] + cum_reward)
                    print("episode:", episode, "score:", cum_reward,  "epsilon:", self.epsilon, "auc:", self.auc[-1])

                    torch.save(self.policy_net, "chain_agent")
