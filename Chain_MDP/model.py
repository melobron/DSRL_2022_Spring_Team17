import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_states=10, n_actions=2, n_feats=64):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, n_feats),
            nn.ReLU(),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(),
            nn.Linear(n_feats, n_actions)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    input = torch.Tensor(1)
    model = DQN()
    output = model(input)
    print(output.shape)
