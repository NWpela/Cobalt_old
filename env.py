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
    def __init__(self, cash_euro: float, traded_assets: list, state_data_proportion: float = 1/300,
                 nb_cut_alpha2: int = 4, interval: str = "15m", inventory_rate: float = 0.005,
                 fee_rate: float = 0.001, rel_var_limit_min: float = 0.005, MtM_proportion_lim: float = 0.5,
                 min_price_value: float = 1e-5):
        self.traded_assets = traded_assets
        self.N_assets = len(traded_assets)

        # define asset manager and data manager
        self.asset_manager = AssetManager(cash_euro, traded_assets, inventory_rate=inventory_rate, fee_rate=fee_rate,
                                          rel_var_limit_min=rel_var_limit_min, MtM_proportion_lim=MtM_proportion_lim,
                                          min_price_value=min_price_value)
        self.data_manager = DataManager(traded_assets, state_data_proportion=state_data_proportion,
                                        nb_cut_alpha2=nb_cut_alpha2, interval=interval)
        self.rewards = []

    def get_state_dim(self) -> int:
        return self.data_manager.get_state_data_dim() + self.asset_manager.get_state_data_dim()

    def get_action_dim(self) -> int:
        return self.asset_manager.get_action_data_dim()

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
    def next_step_market(self) -> bool:
        """
            Computes the new market data
        """
        return self.data_manager.next_step()

    # 2
    def get_state(self) -> list:
        """
            Return the current state of the environment as a list for the model (after the new market data)
        """
        state_market = self.data_manager.get_market_state_data()
        # prices are both ask and bid prices because there is no difference in the current model
        prices = self.data_manager.get_spot_prices()
        spot_ref = self.data_manager.dict_spot_ref
        self.asset_manager.execute_all_possible_orders(prices, prices)
        state_asset = self.asset_manager.get_all_current_state_data(prices, prices, spot_ref)

        return state_market + state_asset

    # 3
    def get_reward(self) -> float:
        """
             Returns the reward as a float add it to the rewards list
        """
        reward = self.asset_manager.reward
        self.rewards.append(reward)
        return reward

    # 4
    def do_action(self, action_list: np.array):
        """
            Do the action represented by the action_list list (each component in [-1, 1])
        """
        # transforms the vector to get values in [0, 1]
        action_list = 0.5 * (action_list + 1)

        tot_brut_prop = action_list[:self.N_assets].sum()
        if tot_brut_prop > 0:
            buy_proportion_euro = action_list[:self.N_assets] / tot_brut_prop
        else:
            buy_proportion_euro = np.array([1/self.N_assets for _ in range(self.N_assets)])
        free_cash_quantity = action_list[self.N_assets]
        # multiplying by (1 - self.rel_var_limit_min) to avoid the value 1
        relative_price_change_buy = action_list[self.N_assets+1:2*self.N_assets+1] *\
                                    (1 - self.asset_manager.rel_var_limit_min)
        free_asset_quantity = action_list[2*self.N_assets+1:3*self.N_assets+1]
        # multiplying by (1 - self.rel_var_limit_min) to avoid the value 1
        relative_price_change_sell = action_list[3*self.N_assets+1:] *\
                                    (1 - self.asset_manager.rel_var_limit_min)

        spot_prices = self.data_manager.get_spot_prices()

        self.asset_manager.take_positions(buy_proportion_euro, free_cash_quantity, relative_price_change_buy,
                                          free_asset_quantity, relative_price_change_sell, spot_prices,
                                          spot_prices)

    # 5
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
