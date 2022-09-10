import numpy as np
import pandas as pd
from little_logger import Little_logger
import datetime as dt

logger = Little_logger("AssetManager")


class AssetManager:
    """
        Class to handle the trading resources used by the algorithm
        Handles trades, cash, inventory, etc...
    """

    def __init__(self, cash_euro: float, traded_assets: list, inventory_rate: float, fee_rate: float,
                 MtM_proportion_lim: float):
        # Arguments
        self.traded_assets = traded_assets
        self.inventory_rate = inventory_rate
        self.fee_rate = fee_rate
        self.MtM_proportion_lim = MtM_proportion_lim
        # set the initial cash in memory for reset
        self.initial_cash_euro = cash_euro

        # Performance parameters
        self.previous_cash_euro = cash_euro
        self.cash_euro = cash_euro
        self.previous_MtM = cash_euro
        self.MtM = cash_euro
        self.previous_MtM_inventory = 0
        self.MtM_inventory = 0
        self.PnL = 0
        self.reward = 0

        # Asset objects
        # set trades dataframe
        self.trades = pd.DataFrame(columns=["Time", "Type", "Asset", "Amount_euro", "Amount_asset", "Price"])
        # set inventory initialized with all traded assets
        self.inventory = {asset: 0 for asset in traded_assets}
        self.N_assets = len(traded_assets)

    def reset(self):
        # Performance parameters
        self.previous_cash_euro = self.initial_cash_euro
        self.cash_euro = self.initial_cash_euro
        self.previous_MtM = self.initial_cash_euro
        self.MtM = self.initial_cash_euro
        self.previous_MtM_inventory = 0
        self.MtM_inventory = 0
        self.PnL = 0
        self.reward = 0

        # Asset objects
        # set trades dataframe
        self.trades = pd.DataFrame(columns=["Time", "Type", "Asset", "Amount_euro", "Amount_asset", "Price"])
        # set inventory initialized with all traded assets
        self.inventory = {asset: 0 for asset in self.traded_assets}

    # --- TRADES BASICS ---

    def get_last_trade_id(self) -> int:
        # to get the last order id
        if len(self.trades.index) > 0:
            return max(self.trades.index) + 1
        else:
            return 1

    def add_trade(self, trade: pd.Series or dict):
        # add an order as a series with a name or a dict with auto id handling
        if type(trade) == pd.Series:
            # in this case order needs to have the standard format with an id as name
            self.trades = pd.concat([self.trades, trade.to_frame().T])
        elif type(trade) == dict:
            orderId = self.get_last_trade_id()
            trade = pd.Series(trade, name=orderId)
            self.trades = pd.concat([self.trades, trade.to_frame().T])
        else:
            logger.error(f"Invalid type for order: {type(trade)}")

    def execute_trade(self, trade: dict):
        # Execute the order with specified orderId
        # The order amounts are designed to have signs
        # Process depending on fees handling
        if trade["Type"] == "buy":
            self.cash_euro += trade["Amount_euro"]
            self.inventory[trade["Asset"]] += trade["Amount_asset"] * (1 - self.fee_rate)
        elif trade["Type"] == "sell":
            self.cash_euro += trade["Amount_euro"] * (1 - self.fee_rate)
            self.inventory[trade["Asset"]] += trade["Amount_asset"]
        else:
            logger.error(f"Invalid order type for order execution: {trade['Type']}")
        # trade is then historized
        self.add_trade(trade)

    # --- STATE DATA AND PERFORMANCE METHODS ---

    def get_new_state_and_reward(self, new_spot_prices: dict, spot_ref: dict) -> tuple:
        # Returns the current asset state data for the IA model
        # Also computes the performance values as well as the reward

        # cash
        asset_state_data = [self.cash_euro / self.initial_cash_euro]
        # inventory
        for asset in self.traded_assets:
            asset_state_data.append(self.inventory[asset] * spot_ref[asset] / self.initial_cash_euro)

        # performances refreshing
        self.previous_cash_euro = self.cash_euro
        self.previous_MtM = self.MtM
        self.previous_MtM_inventory = self.MtM_inventory
        self.MtM_inventory = 0
        for asset in self.traded_assets:
            self.MtM_inventory += self.inventory[asset] * new_spot_prices[asset]
        self.MtM = self.cash_euro + self.MtM_inventory
        self.PnL = self.MtM - self.previous_MtM
        # reward
        self.reward = self.PnL - self.inventory_rate * self.MtM_inventory

        return asset_state_data, self.reward

    def get_state_data_dim(self) -> int:
        # returns the length of the list containing the state data
        return len(self.traded_assets) + 1

    def get_action_data_dim(self) -> int:
        # returns the length of the list containing the action data
        return len(self.traded_assets) + 1

    # --- ACTION METHODS ---

    def take_positions(self, used_cash_prop: float, empirical_prop: np.array, spot_prices: dict):
        # Takes the positions with the empirical proportions computed by the algorithm and the spot prices from the data
        # Only uses used_cash_prop proportion of invested cash for buy orders
        # empirical_prop has to contain values in [-1, 1] and used_cash_prop in [0, 1]

        used_cash_euro = used_cash_prop * self.cash_euro

        # compute the total buy
        total_buy_prop = 0
        for prop in empirical_prop:
            if prop > 0:
                total_buy_prop += prop

        for i in range(self.N_assets):
            asset = self.traded_assets[i]
            spot_price = spot_prices[asset]

            # if buy order
            if empirical_prop[i] > 0:
                amount_euro = - empirical_prop[i] / total_buy_prop * used_cash_euro  # < 0
                amount_asset = - amount_euro / spot_price  # > 0

                trade = dict(
                    Time=dt.datetime.now(),
                    Type="buy",
                    Asset=asset,
                    Amount_euro=amount_euro,
                    Amount_asset=amount_asset,
                    Price=spot_price
                )
                self.execute_trade(trade)
            elif empirical_prop[i] < 0:
                amount_asset = empirical_prop[i] * self.inventory[asset]  # < 0
                amount_euro = - amount_asset * spot_price  # > 0

                trade = dict(
                    Time=dt.datetime.now(),
                    Type="sell",
                    Asset=asset,
                    Amount_euro=amount_euro,
                    Amount_asset=amount_asset,
                    Price=spot_price
                )
                self.execute_trade(trade)
            # if empirical_prop[i] == 0 nothing is done

    # --- DONE METHOD ---

    def is_done(self) -> bool:
        return abs(self.MtM - self.initial_cash_euro) > self.MtM_proportion_lim * self.initial_cash_euro

    # --- DEPRECIATED / UNUSED ---
