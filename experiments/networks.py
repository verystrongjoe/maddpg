import torch as th
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):

    # todo : here, batch_first always true
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

    def __init__(self, batch_dim, agent_dim, observation_dim, out_dim, hidden_dim=32, nonlin=F.relu,
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

        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(observation_dim, hidden_dim)
        self.bilstm = nn.LSTM(batch_dim, hidden_dim, num_layers=2,  bidirectional=True)

        self.linear2 = nn.Linear(batch_dim, out_dim)

        self.softmax1 = nn.Softmax

    def forward(self, o):
        h = TimeDistributed(self.linear1)(o)
        h = TimeDistributed(F.relu)(h)
        h1 = self.bilstm(h)
        h = h1[0][:, :, 32:64] + h1[0][:, :, 0:32]
        h = TimeDistributed(self.linear2)(h)
        a = TimeDistributed(self.softmax1)(h)
        # a = self.out(h)

        return a

class CriticNetwork(nn.Module):
    def __init__(self, batch_dim, agent_dim, observation_dim, action_dim, out_dim=1, hidden_dim=32, nonlin=F.relu,
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

        self.nonlin = nonlin
        self.batch_n = batch_dim

        self.td1 = TimeDistributed(nn.Linear(observation_dim+action_dim, hidden_dim))
        self.relu1 = TimeDistributed(nonlin)

        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstm = nn.LSTM(32, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.td3 = TimeDistributed(nn.Linear(hidden_dim, 32))
        self.relu2 = TimeDistributed(nonlin)
        self.out = TimeDistributed(nn.Linear(32, out_dim))

    def forward(self, o, a):
        new_input = th.cat(o, a, dim=1) # aggregate observation and action for each agent to feed into critic network
        h = self.td1(new_input)
        h = TimeDistributed(F.relu(h))(h)
        h = self.bilstm(h)
        h = self.td3(h)
        h = self.relu2(h)
        qv = self.out(h)

        return qv