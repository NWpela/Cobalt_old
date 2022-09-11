import torch
import math as m
import numpy as np
import torch.nn as nn


"""
    This file contains the full model for the Kobalt project
"""


class Kobalt_model_A2C_continuous(nn.Module):
    """
        This class contains the whole neural model for the Kobalt project
        It is used to handle continuous action spaces with 2 outputs:
            - the gaussian probability distribution output (mu, var)
            - as well as the value output
    """
    def __init__(self, input_size: int, action_size: int, hidden_size: int, n_layers_base: int, n_layers_actor: int,
                 n_layers_critic: int, channels_per_asset_alpha4: int, n_assets_alpha4: int, kernel_size_alpha4: int,
                 input_size_alpha4: int):
        super(Kobalt_model_A2C_continuous, self).__init__()

        # standard input parameters
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # alpha4 input parameters
        self.channels_per_asset_alpha4 = channels_per_asset_alpha4  # in fact n_baselines
        self.n_assets_alpha4 = n_assets_alpha4
        self.kernel_size_alpha4 = kernel_size_alpha4
        self.input_size_alpha4 = input_size_alpha4

        # alpha4 layer
        self.alpha4 = self.get_alpha4_input_sequential()

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
        # base sequential is adapted to also be fed with alpha4 network's resulting tensor
        layers_list = []
        i = 0
        while i < n_layers:
            if i == 0:
                layers_list.append(nn.Linear(self.input_size + self.n_assets_alpha4, self.hidden_size))
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

    def get_alpha4_input_sequential(self):
        # Function created to generate the network for the alpha input to the model
        # There are as many outputs as assets (1 value per asset)
        layers_list = []

        # first layer
        n_channels_in = self.channels_per_asset_alpha4 * self.n_assets_alpha4
        layers_list.append(nn.Conv1d(n_channels_in, 2 * self.n_assets_alpha4, self.kernel_size_alpha4))
        out_size = self.input_size_alpha4 - self.kernel_size_alpha4 - 1
        layers_list.append(nn.ReLU())

        # second layer
        n_channels_in = 2 * self.n_assets_alpha4
        layers_list.append(nn.Conv1d(n_channels_in, self.n_assets_alpha4, self.kernel_size_alpha4))
        out_size = out_size - self.kernel_size_alpha4 - 1
        layers_list.append(nn.ReLU())

        # last layer
        last_kernel_size = out_size - 2  # to have only 1 output value per channel
        layers_list.append(nn.Conv1d(self.n_assets_alpha4, self.n_assets_alpha4, last_kernel_size))
        layers_list.append(nn.Tanh())

        return nn.Sequential(*layers_list)

    # --- FORWARD FUNCTION ---

    def forward(self, standard_state: np.array, alpha4_state: np.array):
        standard_state_tensor = torch.FloatTensor(standard_state)
        alpha4_state_tensor = torch.FloatTensor(alpha4_state)

        # alpha4 first
        alpha4_out = self.alpha4(alpha4_state_tensor)
        alpha4_out = torch.squeeze(alpha4_out)

        # concatenate output with standard state input
        base_in = torch.concat([standard_state_tensor, alpha4_out])

        # pass through base
        base_out = self.base(base_in)

        return self.mu(base_out), self.var(base_out), self.value(base_out)


def calc_log_prob(mu_v_n: torch.Tensor, var_v_n: torch.Tensor, actions_v: np.array) -> torch.Tensor:
    actions_t = torch.FloatTensor(actions_v)
    # Returns the log of the normal distribution as a tensor for vector-like inputs
    p1 = - ((mu_v_n - actions_t) ** 2) / (2*var_v_n.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * m.pi * var_v_n))
    return (p1 + p2).mean()
