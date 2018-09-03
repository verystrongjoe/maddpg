import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class ActorNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, agent_dim, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int) : Number of dimensions in input  (agents, observation)
            out_dim (int)   : Number of dimensions in output

            hidden_dim (int) : Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ActorNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)  # input_dim 2 or 3 dimensional
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.nonlin = nonlin

        self.td1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))(input_dim)
        self.td2 = TimeDistributed(nonlin(agent_dim, hidden_dim))(agent_dim, hidden_dim)
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstm = nn.LSTM(32, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.td3 = TimeDistributed(nn.Linear(32, out_dim))(32)
        self.out = TimeDistributed(nn.Softmax(out_dim))(out_dim)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class CriticNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, agent_dim, input_dim, action_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int): Number of dimensions in input  (agents, observation)
            out_dim (int): Number of dimensions in output

            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CriticNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)  # input_dim 2 or 3 dimensional
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.nonlin = nonlin

        self.td1 = TimeDistributed(nn.Linear(input_dim, hidden_dim))(input_dim)
        self.td2 = TimeDistributed(nonlin(agent_dim, hidden_dim))()

        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstm = nn.LSTM(32, hidden_dim + action_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.td3 = TimeDistributed(nn.Linear(32, out_dim))(32)
        self.td3 = TimeDistributed(nonlin(out_dim))(32)
        self.out = TimeDistributed(nn.Linear(out_dim, ))(out_dim)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out