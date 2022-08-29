# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:22:17 2022

@author: Gaëtan Chanet
"""

# Here we start, I hope this will lead me to JP Morgan, anyway, we'll see...
import pandas as pd
import torch
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import math as m
import model
from env import Kobalt_env
from model import Kobalt_model_A2C_continuous
from little_logger import Little_logger

matplotlib.use('Qt5Agg')
pd.set_option('display.max_rows', 50, 'display.max_columns', None)
logger = Little_logger("Kobalt_1")

# Model parameters
HIDDEN_SIZE = 100
N_LAYERS_BASE = 1
N_LAYERS_ACTOR = 2
N_LAYERS_CRITIC = 2
LEARNING_RATE = 3e-4

GAMMA = 0.99
ENTROPY_BETA = 1e-1
STEPS_PER_EPISOD = 50
MAX_EPISODS = 100

# Env parameters
CASH_EURO = 100
TRADED_ASSETS = ["BTC", "ETH", "XRP", "LINK", "XLM", "EOS", "BNB", "TRX", "DOGE", "MATIC", "AVAX", "SOL", "UNI"]
STATE_DATA_PROPORTION = 1/500
NB_CUT_ALPHA2 = 3
INTERVAL = "15m"
INVENTORY_RATE = 1e-5
FEE_RATE = 1e-3
REL_VAR_LIMIT_MIN = 5e-5
MTM_PROPORTION_LIM = 0.5
MIN_PRICE_VALUE = 1e-3


class Kobalt_1:
    def __init__(self):
        logger.info("***** Kobalt Development Edition *****")
        # Environment
        self.env = Kobalt_env(CASH_EURO, TRADED_ASSETS, state_data_proportion=STATE_DATA_PROPORTION,
                              nb_cut_alpha2=NB_CUT_ALPHA2, interval=INTERVAL, inventory_rate=INVENTORY_RATE,
                              fee_rate=FEE_RATE, rel_var_limit_min=REL_VAR_LIMIT_MIN,
                              MtM_proportion_lim=MTM_PROPORTION_LIM, min_price_value=MIN_PRICE_VALUE)
        self.env.reset_all()

        # Model state/action dimensions
        self.state_dim = self.env.get_state_dim()
        self.action_dim = self.env.get_action_dim()

        # Model definition
        self.model = Kobalt_model_A2C_continuous(self.state_dim, self.action_dim, HIDDEN_SIZE, N_LAYERS_BASE,
                                                 N_LAYERS_ACTOR, N_LAYERS_CRITIC, learning_rate=LEARNING_RATE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Global data
        self.training_episods = []
        self.training_rewards = []
        self.training_MtM = []

    def train(self):
        logger.info("--- Training started ---")
        # self.env.reset_all()
        self.training_rewards = []
        next_step_possible = self.env.is_next_step_possible()
        state = self.env.get_state()

        episod_num = 0
        while episod_num < MAX_EPISODS and next_step_possible:
            # reset only positions for each episod
            self.env.reset_positions()
            self.training_episods.append(episod_num)

            log_probs_n = []
            values_n = []
            values = []
            rewards = []
            entropy_term = torch.FloatTensor([0])

            step_num_episod = 0
            done = self.env.is_done()
            while step_num_episod < STEPS_PER_EPISOD and next_step_possible and not done:

                # variables computed from the model
                mu_v_n, var_v_n, value_n = self.model.forward(state)
                value = value_n.detach().numpy()
                mu_v = mu_v_n.detach().numpy()
                sigma_v = torch.sqrt(var_v_n).data.cpu().numpy()

                # action and entropy
                action_v = np.random.normal(mu_v, sigma_v)
                # compute log_prob as a tensor number
                log_prob_n = model.calc_log_prob(mu_v_n, var_v_n, action_v)
                # /!\ this entropy term aims to work with tensor of steps data (here vector of action data)
                entropy_term += (-(torch.log(2*m.pi*var_v_n) + 1)/2).mean()

                # new state and reward
                # action argument for env is transformed to fit into the [-1, 1] interval (to be reviewed)
                env_action = action_v.clip(-1, 1)
                self.env.do_action(env_action)
                self.env.next_step_market()
                new_state = self.env.get_state()
                reward = self.env.get_reward()

                # adding variables to lists
                rewards.append(reward)
                values.append(value)
                values_n.append(value_n)
                log_probs_n.append(log_prob_n)
                state = new_state

                # computing next step for market data and recovering done information
                next_step_possible = self.env.is_next_step_possible()
                done = self.env.is_done()
                step_num_episod += 1

            print(self.env.asset_manager.orders)
            print(self.env.asset_manager.trades)

            # when the episod is finished
            _, _, Qval = self.model.forward(state)
            Qval = Qval.detach().numpy()[0]
            self.training_rewards.append(np.sum(rewards))
            MtM = self.env.asset_manager.MtM
            self.training_MtM.append(MtM)
            if episod_num % 1 == 0:
                logger.info("Episode: {}, Steps: {}, Reward: {}, MtM: {} \n".format(episod_num, step_num_episod,
                                                                                    np.sum(rewards), MtM))

            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval

            # compute tensors
            values = torch.FloatTensor(np.array(values))
            Qvals = torch.FloatTensor(Qvals)
            log_probs_n = torch.stack(log_probs_n)

            # advantage and loss
            advantage = Qvals - values
            actor_loss = (-log_probs_n * advantage).mean()
            values_n = torch.FloatTensor(values_n)
            critic_loss = 0.5 * (Qvals - values_n).pow(2).mean()
            ac_loss = actor_loss + critic_loss + ENTROPY_BETA * entropy_term

            # grad
            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()

            episod_num += 1

        logger.info("--- Training finished ---")

    def training_plot(self):
        plt.plot(self.training_episods, self.training_rewards, self.training_episods, self.training_MtM)


if __name__ == "__main__":
    kob = Kobalt_1()
