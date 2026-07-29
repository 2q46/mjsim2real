import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class MLP(nn.Module):

    def __init__(self, obs_dim: float, n_actions: float):

        super(MLP, self).__init__()

        self.fcl1 = nn.Linear(in_features=obs_dim, out_features=64)
        self.fcl2 = nn.Linear(in_features=64, out_features=32)
        self.fcl3 = nn.Linear(in_features=32, out_features=n_actions)


    def forward(self, x: torch.Tensor):

        x = self.fcl1(x)
        x = F.tanh(x)
        x = self.fcl2(x)
        x = F.relu(x)
        x = self.fcl3(x)
                