import numpy as np

from assetManager import AssetManager
from dataManager import DataManager


"""
    This file contains the full environment for the Kobalt project
"""

class Kobalt_env:
    """
        Kobalt env should be used as follows:
            0: use reset to reset the parameters
            1 -> 2 -> 3 -> 4 -> 1 -> ...
    """
    def __init__(self, cash_euro: float, traded_assets: list, alpha4_ema_win_list: list,
                 state_data_proportion: float = 1/300, interval: str = "15m", inventory_rate: float = 0.005,
                 fee_rate: float = 0.001, MtM_proportion_lim: float = 0.5):
        self.traded_assets = traded_assets
        self.N_assets = len(traded_assets)

        # define asset manager and data manager
        self.asset_manager = AssetManager(cash_euro, traded_assets, inventory_rate=inventory_rate, fee_rate=fee_rate,
                                          MtM_proportion_lim=MtM_proportion_lim)
        self.data_manager = DataManager(traded_assets, alpha4_ema_win_list, state_data_proportion=state_data_proportion,
                                        interval=interval)
        self.N_baselines = self.data_manager.N_baselines
        self.spot_prices = None
        self.rewards = []

    def get_state_dim(self) -> int:
        return self.data_manager.get_state_data_dim() + self.asset_manager.get_state_data_dim()

    def get_action_dim(self) -> int:
        return self.asset_manager.get_action_data_dim()

    # 0
    def reset_all(self):
        """
            Reset all components of the environment, in particular the data
        """
        self.asset_manager.reset()
        self.data_manager.reset()

    def reset_positions(self):
        """
            Reset the asset manager but keep the current market state
        """
        self.asset_manager.reset()

    # 1
    def next_step_market(self):
        """
            Computes the new market data
        """
        self.data_manager.next_step()
        self.spot_prices = self.data_manager.get_spot_prices()

    # 2
    def get_new_state_and_reward(self) -> tuple:
        """
            Return the new state of the environment as a list and the reward
        """
        # market
        state_market = self.data_manager.get_market_state_data()

        # alpha4
        state_alpha4 = self.data_manager.get_alpha4_state_data()

        # assets
        spot_ref = self.data_manager.dict_spot_ref
        state_asset, reward = self.asset_manager.get_new_state_and_reward(self.spot_prices, spot_ref)
        self.rewards.append(reward)

        return np.array(state_market + state_asset), state_alpha4, reward

    # 3
    def do_action(self, action_array: np.array):
        """
            Do the action represented by the action_list list (each component in [-1, 1])
        """

        used_cash_prop = (action_array[0] + 1) / 2  # because this is a proportion in [0, 1]
        empirical_prop = action_array[1:]
        self.asset_manager.take_positions(used_cash_prop, empirical_prop, self.spot_prices)

    # 4
    def is_next_step_possible(self):
        """
            Indicates if a next step is possible given the market data used
        """
        return self.data_manager.is_next_step_possible()

    def is_done(self):
        """
            Indicates if the simulation is finished
        """
        return self.asset_manager.is_done()
