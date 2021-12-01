import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class Q_ConvNet(nn.Module):
    def __init__(self, in_channels, num_actions, dueling=True, noisy=True, distributional=True, atom_size=51, v_min=-10.0, v_max=10.0):
        super(Q_ConvNet, self).__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.dueling = dueling
        self.noisy = noisy
        self.distributional = distributional
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to("cpu")

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        if self.dueling == False:
            if self.noisy == False:
                self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
                if self.distributional == False:
                    self.output = nn.Linear(in_features=128, out_features=num_actions)
                else:
                    self.output = nn.Linear(in_features=128, out_features=num_actions * self.atom_size)
            else:
                self.fc_hidden = NoisyLinear(in_features=num_linear_units, out_features=128)
                if self.distributional == False:
                    self.output = NoisyLinear(in_features=128, out_features=num_actions)
                else:
                    self.output = NoisyLinear(in_features=128, out_features=num_actions * self.atom_size)

        else:
            if self.noisy == False:
                self.value_fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
                self.value_output = nn.Linear(in_features=128, out_features=1)
                self.advantage_fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
                if self.distributional == False:
                    self.advantage_output = nn.Linear(in_features=128, out_features=num_actions)
                else:
                    self.advantage_output = nn.Linear(in_features=128, out_features=num_actions * self.atom_size)

            else:
                self.value_fc_hidden = NoisyLinear(in_features=num_linear_units, out_features=128)
                self.value_output = NoisyLinear(in_features=128, out_features=1)
                self.advantage_fc_hidden = NoisyLinear(in_features=num_linear_units, out_features=128)
                self.advantage_output = NoisyLinear(in_features=128, out_features=num_actions)
                if self.distributional == False:
                    self.advantage_output = NoisyLinear(in_features=128, out_features=num_actions)
                else:
                    self.advantage_output = NoisyLinear(in_features=128, out_features=num_actions * self.atom_size)


    def forward(self, x):
        if self.dueling == False:
            x = F.relu(self.conv(x))
            x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
            if self.distributional == False:
                return self.output(x)
            else:
                q_atoms = self.output(x).view(-1, self.num_actions, self.atom_size)
                dist = F.softmax(q_atoms, dim=-1)
                dist = dist.clamp(min=1e-3)
                q = torch.sum(dist * self.support, dim=2)
                return q
        else:
            x = F.relu(self.conv(x))
            advantage = F.relu(self.advantage_fc_hidden(x.view(x.size(0), -1)))
            value = F.relu(self.value_fc_hidden(x.view(x.size(0), -1)))

            advantage = self.advantage_output(advantage)
            value = self.value_output(value)
            if self.distributional == False:
                return value + advantage - advantage.mean(dim=-1, keepdim=True)
            else:
                q_atoms = advantage.view(-1, self.num_actions, self.atom_size)
                dist = F.softmax(q_atoms, dim=-1)
                dist = dist.clamp(min=1e-3)
                q = torch.sum(dist * self.support, dim=2)
                return value + q - q.mean(dim=-1, keepdim=True)

    def reset_noisy(self):
        if self.noisy==True:
            if self.dueling==False:
                self.fc_hidden.reset_noise()
                self.output.reset_noise()
            else:
                self.advantage_fc_hidden.reset_noise()
                self.advantage_output.reset_noise()
                self.value_fc_hidden.reset_noise()
                self.value_output.reset_noise()

    def distribution(self, x):
        q_atoms = self.forward(x).view(-1, self.num_actions, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.input = in_features
        self.output = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.input)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.output)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.input)
        epsilon_out = self.scale_noise(self.output)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size):
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())




class Ensemble_Q_ConvNet(nn.Module):
    def __init__(self, in_channels, num_actions, ensemble_size):
        super(Ensemble_Q_ConvNet, self).__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.ensemble_size = ensemble_size

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.ensemble_fc_hiddens = []
        self.ensemble_outputs = []
        for head in range(ensemble_size):
            self.ensemble_fc_hiddens.append(nn.Linear(in_features=num_linear_units, out_features=128))
            self.ensemble_outputs.append(nn.Linear(in_features=128, out_features=num_actions))

    def forward(self, x, head_idx=None):
        x = F.relu(self.conv(x))
        if head_idx==None:
            head_idx = np.random.randint(self.ensemble_size)
        x = F.relu(self.ensemble_fc_hiddens[head_idx](x.view(x.size(0), -1)))
        return self.ensemble_outputs[head_idx](x)
