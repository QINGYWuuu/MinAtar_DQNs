import torch.nn as nn
import torch.nn.functional as F

class Q_Net(nn.Module):
    def __init__(self, obs_dim, act_dim, mid_dim=24):
        super(Q_Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, act_dim)
        )
    def forward(self, state):
        return self.layers(state)


class Q_ConvNet(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Q_ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=num_actions)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)
