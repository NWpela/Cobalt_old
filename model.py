import torch
import math as m
import numpy as np
import torch.nn as nn


"""
    This file contains the full model for the Cobalt project
"""


class Cobalt_model_A2C_continuous(nn.Module):
    """
        This class contains the whole neural model for the Cobalt project
        It is used to handle continuous action spaces with 2 outputs:
            - the gaussian probability distribution output (mu, var)
            - as well as the value output
    """
    def __init__(self, input_size: int, action_size: int, hidden_size: int, n_layers_base: int, n_layers_actor: int,
                 n_layers_critic: int):
        super(Cobalt_model_A2C_continuous, self).__init__()

        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # base layer (will probably be removed)
        self.base = self.__get_base_sequential(n_layers_base)

        # mu layer
        self.mu = self.__get_mu_sequential(n_layers_actor)

        # var layer
        self.var = self.__get_var_sequential(n_layers_actor)

        # value layer
        self.value = self.__get_value_sequential(n_layers_critic)

    # --- AUTO BUILDING FUNCTIONS ---

    def __get_base_sequential(self, n_layers: int):
        layers_list = []
        i = 0
        while i < n_layers:
            if i == 0:
                layers_list.append(nn.Linear(self.input_size, self.hidden_size))
                layers_list.append(nn.ReLU())
            else:
                layers_list.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers_list.append(nn.ReLU())
            i += 1
        return nn.Sequential(*layers_list)

    def __get_mu_sequential(self, n_layers: int):
        layers_list = []
        i = 0
        while i < n_layers:
            if i == n_layers - 1:
                layers_list.append(nn.Linear(self.hidden_size, self.action_size))
                layers_list.append(nn.Tanh())
            else:
                layers_list.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers_list.append(nn.ReLU())
            i += 1
        return nn.Sequential(*layers_list)

    def __get_var_sequential(self, n_layers: int):
        layers_list = []
        i = 0
        while i < n_layers:
            if i == n_layers - 1:
                layers_list.append(nn.Linear(self.hidden_size, self.action_size))
                layers_list.append(nn.Softplus())
            else:
                layers_list.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers_list.append(nn.ReLU())
            i += 1
        return nn.Sequential(*layers_list)

    def __get_value_sequential(self, n_layers: int):
        layers_list = []
        i = 0
        while i < n_layers:
            if i == n_layers - 1:
                layers_list.append(nn.Linear(self.hidden_size, 1))
                layers_list.append(nn.Tanh())
            else:
                layers_list.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers_list.append(nn.ReLU())
            i += 1
        return nn.Sequential(*layers_list)

    # --- FORWARD FUNCTION ---

    def forward(self, state: np.array):
        state_tensor = torch.FloatTensor(state)
        base_out = self.base(state_tensor)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


def calc_log_prob(mu_v_n: torch.Tensor, var_v_n: torch.Tensor, actions_v: np.array) -> torch.Tensor:
    actions_t = torch.FloatTensor(actions_v)
    # Returns the log of the normal distribution as a tensor for vector-like inputs
    p1 = - ((mu_v_n - actions_t) ** 2) / (2*var_v_n.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * m.pi * var_v_n))
    return (p1 + p2).mean()
