from .base_critic import BaseCritic
from torch import nn
from torch import optim
import numpy as np
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import sac_utils
import torch

class SACCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super(SACCritic, self).__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.gamma = hparams['gamma']
        self.Q1 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q2 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q1.to(ptu.device)
        self.Q2.to(ptu.device)
        self.loss = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        # TODO: return the two q values
        # print(obs.size())
        # print(action.size())
        # print(self.ob_dim)
        # print(self.ac_dim)
        tuple = torch.cat([obs.T,action.T]).T
        value1 = self.Q1(tuple)
        value2 = self.Q2(tuple)
        values = torch.cat([value1, value2],1)
        # print(values)
        # print(values.size())

        # Just to be specific, we are expected for the values to be a torch tensor
        # that is [batch_size by 2] where the two comes from the two seperate Q-values 

        return values



        