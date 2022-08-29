import numpy as np
import pandas as pd
from little_logger import Little_logger
import datetime as dt

logger = Little_logger("AssetManager")


class AssetManager:
    """
        Class to handle the trading resources used
        Handles trades, cash, inventory, etc, ...
    """

    def __init__(self, cash_euro: float, traded_assets: list, inventory_rate: float, fee_rate: float,
                 rel_var_limit_min: float, MtM_proportion_lim: float, min_price_value: float):
        # Arguments
        self.traded_assets = traded_assets
        self.inventory_rate = inventory_rate
        self.fee_rate = fee_rate
        self.rel_var_limit_min = rel_var_limit_min
        self.MtM_proportion_lim = MtM_proportion_lim
        self.min_price_value = min_price_value
        # set the initial cash in memory for reset
        self.initial_cash_euro = cash_euro

        # free amounts
        self.free_cash_euro = cash_euro
        self.dict_free_amounts_asset = {asset: 0 for asset in traded_assets}

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
        # set orders and trades dataframes
        self.orders = pd.DataFrame(columns=["Time", "Type", "Asset", "Amount_euro", "Amount_asset", "Price"])
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

        # free amounts
        self.free_cash_euro = self.initial_cash_euro
        self.dict_free_amounts_asset = {asset: 0 for asset in self.traded_assets}

        # Asset objects
        # set orders and trades dataframes
        self.orders = pd.DataFrame(columns=["Time", "Type", "Asset", "Amount_euro", "Amount_asset", "Price"])
        self.trades = pd.DataFrame(columns=["Time", "Type", "Asset", "Amount_euro", "Amount_asset", "Price"])
        # set inventory initialized with all traded assets
        self.inventory = {asset: 0 for asset in self.traded_assets}

    # --- ORDERS AND TRADES BASICS ---

    def get_last_order_id(self) -> int:
        # to get the last order id
        if len(self.orders.index) > 0:
            return max(self.orders.index) + 1
        else:
            return 1

    def add_order(self, order: pd.Series or dict):
        # add an order as a series with a name or a dict with auto id handling
        if type(order) == pd.Series:
            # in this case order needs to have the standard format with an id as name
            self.orders = pd.concat([self.orders, order.to_frame().T])
        elif type(order) == dict:
            orderId = self.get_last_order_id()
            order = pd.Series(order, name=orderId)
            self.orders = pd.concat([self.orders, order.to_frame().T])
        else:
            logger.error(f"Invalid type for order: {type(order)}")

        # free amounts handling
        if order.Type == "buy":
            self.free_cash_euro += order.Amount_euro
        elif order.Type == "sell":
            self.dict_free_amounts_asset[order.Asset] += order.Amount_asset
        else:
            logger.error(f"Invalid order type to handle free amounts: {order.Type}")

        #logger.info(f"Order added: {order.Type} {abs(order.Amount_asset)} {order.Asset} for {abs(order.Amount_euro)}€, "
        #            f"price: {order.Price}")

    def remove_order_with_Id(self, orderId: int):
        # remove the order with orderId specified
        if orderId in self.orders.index:
            self.orders.drop(index=orderId, inplace=True)
        else:
            logger.error(f"No such order id to remove: {orderId}")

    # --- ORDERS AND TRADES MAIN FUNCTIONS ---

    def historize_order_with_Id(self, orderId: int):
        # put the order with id orderId as executed trade
        if orderId in self.orders.index:
            order = self.orders.loc[orderId]
            self.remove_order_with_Id(orderId)
            self.trades = pd.concat([self.trades, order.to_frame().T])
        else:
            logger.error(f"No such order id to historize: {orderId}")

    def execute_order_with_id(self, orderId: int):
        # execute the order with specified orderId
        # the order amounts are designed to have signs
        if orderId in self.orders.index:
            order = self.orders.loc[orderId]
            # process depending on fees handling
            if order.Type == "buy":
                self.cash_euro += order["Amount_euro"]
                self.inventory[order["Asset"]] += order["Amount_asset"] * (1 - self.fee_rate)
                # free cash refreshing
                self.dict_free_amounts_asset[order["Asset"]] += order["Amount_asset"]
            elif order.Type == "sell":
                self.cash_euro += order["Amount_euro"] * (1 - self.fee_rate)
                self.inventory[order["Asset"]] += order["Amount_asset"]
                # free amount refreshing
                self.free_cash_euro += order["Amount_euro"]
            else:
                logger.error(f"Invalid order type for order execution: {order.Type}")
            self.historize_order_with_Id(orderId)
        else:
            logger.error(f"No such order id to execute: {orderId}")

    def get_executable_sell_orders_ids(self, ask_prices: dict) -> list:
        # bid_prices needs to have the format {"_asset1_": _bid_price1_, "_asset2_": _bid_price2_, ...}

        # creates a copy of order to add another field indicating if the order should be executed
        orders = self.orders[self.orders.Type == "sell"].copy()

        for asset in ask_prices.keys():
            mask_asset = self.orders.Asset == asset
            ask_price = ask_prices[asset]
            orders.loc[mask_asset, "Executed"] = orders.loc[mask_asset, "Price"] <= ask_price
        executed_sell_orders = orders[orders.Executed]

        return list(executed_sell_orders.index)

    def get_executable_buy_orders_ids(self, bid_prices: dict) -> list:
        # bid_prices needs to have the format {"_asset1_": _bid_price1_, "_asset2_": _bid_price2_, ...}

        # creates a copy of order to add another field indicating if the order should be executed
        orders = self.orders[self.orders.Type == "buy"].copy()

        for asset in bid_prices.keys():
            mask_asset = self.orders.Asset == asset
            bid_price = bid_prices[asset]
            orders.loc[mask_asset, "Executed"] = orders.loc[mask_asset, "Price"] >= bid_price
        executed_buy_orders = orders[orders.Executed]

        return list(executed_buy_orders.index)

    def execute_all_possible_orders(self, ask_prices: dict, bid_prices: dict):
        # executes all possible booked orders and refresh cash and MtM
        all_executable_orders = self.get_executable_sell_orders_ids(ask_prices) + self.get_executable_buy_orders_ids(bid_prices)

        # previous cash and MtM values stored
        self.previous_cash_euro = self.cash_euro
        self.previous_MtM_inventory = self.MtM_inventory
        self.previous_MtM = self.MtM

        # Warning: this part will have to be changed when using API for real time trading
        # Indeed, some orders could be executed and somme others not even if the price is ok
        # -> a new verification will have to be added
        for orderId in all_executable_orders:
            # the current cash is modified here
            self.execute_order_with_id(orderId)

        # MtM, PnL and reward computation
        self.MtM_inventory = 0
        mid_prices = {asset: (ask_prices[asset] + bid_prices[asset]) / 2 for asset in self.traded_assets}
        for asset in self.traded_assets:
            self.MtM_inventory += mid_prices[asset] * self.inventory[asset]
        self.MtM = self.MtM_inventory + self.cash_euro
        self.PnL = self.MtM - self.previous_MtM
        self.reward = self.PnL - self.inventory_rate * self.cash_euro

    # --- GET STATE DATA METHODS ---

    @staticmethod
    def synthesize_positions(df_pos: pd.DataFrame, price: float, position_types: str) -> list:
        """
            To synthesize the positions as an element of state for the environment
            we choose to use mean, var and kurtosis, this model will certainly evolve
        """

        if price != 0:
            if not df_pos.empty:
                # prices are transformed into relative prices from spot
                if position_types == "sell":
                    df_pos.Price = (df_pos.Price - price) / price
                elif position_types == "buy":
                    df_pos.Price = (price - df_pos.Price) / price
                else:
                    logger.error(f"Invalid position_types argument: {position_types}")

                total_asset = df_pos.Amount_asset.sum()
                if total_asset != 0:
                    # get mean
                    m = 0
                    for i, row in df_pos.iterrows():
                        m += row.Price * row.Amount_asset
                    m /= total_asset

                    # get var
                    var = 0
                    for i, row in df_pos.iterrows():
                        var += (row.Price - m)**2 * row.Amount_asset
                    var /= total_asset

                    # get kurtosis
                    kurt = 0
                    for i, row in df_pos.iterrows():
                        kurt += (row.Price - m)**4 * row.Amount_asset
                    kurt /= total_asset

                    if max([m, var, kurt]) > 1000:
                        print(f"{[m, var, kurt]}")
                        print(f"df_pos:{df_pos}")
                        print(f"price:{price}")
                        print(f"position type:{position_types}")
                    return [m, var, kurt]
                else:
                    return [0, 0, 0]
            else:
                return [0, 0, 0]
        else:
            return [0, 0, 0]

    def get_orders_state_data(self, ask_prices: dict, bid_prices: dict) -> list:
        orders_state_data_list = []
        # for each asset traded
        for asset in self.traded_assets:
            # buy orders
            df_buy = self.orders[(self.orders.Type == "buy") & (self.orders.Asset == asset)].copy()
            orders_state_data_list += self.synthesize_positions(df_buy, bid_prices[asset], "buy")
            # sell orders
            df_sell = self.orders[(self.orders.Type == "sell") & (self.orders.Asset == asset)].copy()
            orders_state_data_list += self.synthesize_positions(df_sell, ask_prices[asset], "sell")
        return orders_state_data_list

    def get_all_current_state_data(self, ask_prices: dict, bid_prices: dict, spot_ref: dict) -> list:
        # returns the current asset state data for the IA model
        logger.set_start()

        # cash
        asset_state_data = [self.cash_euro / self.initial_cash_euro]
        # orders
        asset_state_data += self.get_orders_state_data(ask_prices, bid_prices)
        # inventory
        inventory_values = []
        for asset in self.traded_assets:
            inventory_values.append(self.inventory[asset] * spot_ref[asset] / self.initial_cash_euro)
        asset_state_data += inventory_values

        #logger.info("Asset state data computed", with_time=True)
        return asset_state_data

    def get_state_data_dim(self) -> int:
        # returns the length of the list containing the state data
        return 7 * len(self.traded_assets) + 1

    def get_action_data_dim(self) -> int:
        # returns the length of the list containing the action data
        return 4 * len(self.traded_assets) + 1

    # --- ACTION METHODS ---

    def take_positions(self, buy_proportion_euro: np.array, free_cash_quantity: float,
                       relative_spot_change_buy: np.array, free_asset_quantity: np.array,
                       relative_spot_change_sell: np.array, spot_prices_bid: dict, spot_prices_ask: dict):
        # get each asset contribution of free_cash_quantity
        free_cash_quantity_contributions = buy_proportion_euro * free_cash_quantity

        for i in range(self.N_assets):
            asset = self.traded_assets[i]

            # buy order part
            relative_spot_change_buy_value = relative_spot_change_buy[i]
            if self.rel_var_limit_min <= relative_spot_change_buy_value:
                # order added only if the relative variation is enabled (this feature will possibly change)
                price_buy = spot_prices_bid[asset] * (1 - relative_spot_change_buy_value)
                amount_euro_buy = - free_cash_quantity_contributions[i] * self.free_cash_euro
                if - amount_euro_buy >= self.min_price_value:
                    amount_asset_buy = - amount_euro_buy / price_buy

                    buy_order = dict(
                        Time=dt.datetime.now(),
                        Type="buy",
                        Asset=asset,
                        Amount_euro=amount_euro_buy,
                        Amount_asset=amount_asset_buy,
                        Price=price_buy
                    )
                    self.add_order(buy_order)
            #    else:
            #        logger.warning(f"Price too small to add buy order for {asset}: {amount_euro_buy}")
            #else:
            #    logger.warning(f"Relative price variation too small to add buy order for {asset}: "
            #                   f"{relative_spot_change_buy_value}")

            # sell order part
            relative_spot_change_sell_value = relative_spot_change_sell[i]
            if self.rel_var_limit_min <= relative_spot_change_sell_value:
                # order added only if the relative variation is enabled (this feature will possibly change)
                price_sell = spot_prices_ask[asset] * (1 + relative_spot_change_sell_value)
                amount_asset_sell = - free_asset_quantity[i] * self.dict_free_amounts_asset[asset]
                amount_euro_sell = - amount_asset_sell * price_sell
                if amount_euro_sell >= self.min_price_value:
                    sell_order = dict(
                        Time=dt.datetime.now(),
                        Type="sell",
                        Asset=asset,
                        Amount_euro=amount_euro_sell,
                        Amount_asset=amount_asset_sell,
                        Price=price_sell
                    )
                    self.add_order(sell_order)
            #    else:
            #        logger.warning(f"Price too small to add sell order for {asset}: {amount_euro_sell}")
            #else:
            #    logger.warning(f"Relative price variation too small to add sell order for {asset}: "
            #                   f"{relative_spot_change_sell_value}")

    # --- DONE METHOD ---
    def is_done(self) -> bool:
        return abs(self.MtM - self.initial_cash_euro) > self.MtM_proportion_lim * self.initial_cash_euro

    # --- DEPRECIATED / UNUSED ---

    def is_order_possible(self, order: pd.Series) -> bool:
        # function returning true if the order is possible else false
        if order.Type == "buy":
            total_buy_orders_amount_euro = self.orders[self.orders.Type == "buy"].Amount_euro.sum()
            return self.cash_euro + order.Amount_euro + total_buy_orders_amount_euro >= 0
        elif order.Type == "sell":
            asset = order.Asset
            total_sell_orders_amount_asset = self.orders[(self.orders.Type == "sell") & (self.orders.Asset == asset)].Amount_euro.sum()
            return self.inventory[asset] + order.Amount_asset + total_sell_orders_amount_asset >= 0
        else:
            logger.error("Invalid order to check")

    def add_trade(self, trade: pd.Series):
        # trade needs to have the standard format with an id as name
        self.trades = pd.concat([self.trades, trade.to_frame().T])

    def get_adaptative_order(self, order: pd.Series) -> pd.Series:
        # function transforming an order into a possible order
        new_order = order.copy()
        if order.Type == "buy":
            total_buy_orders_amount_euro = self.orders[self.orders.Type == "buy"].Amount_euro.sum()
            new_amount_euro = max(-(self.cash_euro + total_buy_orders_amount_euro), order.Amount_euro)
            if order.Price < self.min_price_value:
                order.Price = self.min_price_value
            new_amount_asset = - (new_amount_euro / order.Price)
            new_order.Amount_euro = new_amount_euro
            new_order.Amount_asset = new_amount_asset
            return new_order
        elif order.Type == "sell":
            asset = order.Asset
            total_sell_orders_amount_asset = self.orders[(self.orders.Type == "sell") & (self.orders.Asset == asset)].Amount_euro.sum()
            new_amount_asset = max(-(self.inventory[asset] + total_sell_orders_amount_asset), order.Amount_asset)
            new_amount_euro = - (new_amount_asset * order.Price)
            new_order.Amount_asset = new_amount_asset
            new_order.Amount_euro = new_amount_euro
            return new_order
        else:
            logger.error("Invalid order to check")

    def take_positions_old(self, amounts: list, rel_prices_var: list, spot_prices: dict):
        # takes the positions with the amounts and the relative price variations in argument
        for i in range(self.N_assets):
            rel_price_var = rel_prices_var[i]
            asset = self.traded_assets[i]
            if abs(rel_price_var) > self.rel_var_limit_min:
                # order added only if the relative variation is enabled (this feature will possibly change)
                amount_asset = amounts[i]
                spot_price = spot_prices[asset]
                price = spot_price * (1 + rel_price_var)
                amount_euro = - amount_asset * price
                order_type = {True: "sell", False: "buy"}[amount_euro > 0]

                order = dict(
                    Time=dt.datetime.now(),
                    Type=order_type,
                    Asset=asset,
                    Amount_euro=amount_euro,
                    Amount_asset=amount_asset,
                    Price=price
                )
                self.add_order(order)
            else:
                logger.warning(f"Relative price variation too small to add order for {asset}: {rel_price_var}")