import pandas as pd
import numpy as np
from pathlib import Path
import os
from little_logger import Little_logger
from indicators import compute_MACD, compute_RSI, compute_EOM, compute_BBANDS

logger = Little_logger("DataManager")


class DataManager:
    """
        Based on a file and load it to get and the required data
        Batch the data to return good parameters for Kobalt (data for episod)
        Now based on binance data via API call
    """

    def __init__(self, assets: list, state_data_proportion: float = 1/300, nb_cut_alpha2: int = 4, interval: str = "15m"):
        self.assets = assets
        self.state_data_proportion = state_data_proportion
        self.nb_cut_alpha2 = nb_cut_alpha2
        self.interval = interval

        self.binance_data_folder = Path("binance_data")

        self.state_data_size = None
        self.dict_raw_data = dict()
        self.dict_state_raw_data = dict()
        self.N_steps = None
        self.step = 1
        self.indicator_columns = ["MACD", "RSI", "EOM", "BBANDS_PERCENT", "BBANDS_RANGE"]

        N_assets = len(assets)
        self.i_series_start = pd.Series(data=[-1 for _ in range(N_assets)], index=assets)
        self.i_series_end = pd.Series(data=[-1 for _ in range(N_assets)], index=assets)

        # References for normalisation
        self.dict_spot_ref = dict()
        self.dict_volume_ref = dict()
        self.dict_spot_diff_std_ref = dict()

    def reset(self):
        # Function used to initialize or reinitialize the data manager with the initial market data
        self.load_all_data_files()
        self.cut_at_same_end()
        self.initialize_i_and_state_parameters()
        self.compute_ref_values()
        self.set_state_data()

    # --- INITIALIZING FUNCTIONS---

    @staticmethod
    def compute_indicators(df: pd.DataFrame):
        compute_MACD(df)
        compute_RSI(df)
        compute_EOM(df)
        compute_BBANDS(df)

    def load_data_file(self, asset: str, version: str = "v1"):
        pair = asset + "EUR"
        file_name = '_'.join([pair, self.interval, version]) + ".csv"
        # load a specific file in the data folder with good parameters
        if os.path.exists(self.binance_data_folder / file_name):
            raw_df = pd.read_csv(self.binance_data_folder / file_name, sep=';')
            self.compute_indicators(raw_df)
            self.dict_raw_data[asset] = raw_df
            logger.info(f"{pair} data successfully loaded")
        else:
            logger.error(f"Impossible to load data for {pair}: {file_name} doesn't exist")

    def load_all_data_files(self, version: str = "v1"):
        for asset in self.assets:
            self.load_data_file(asset, version=version)

    def cut_at_same_end(self):
        # function to set the same end for all data loaded
        end_nums = []
        for raw_data in self.dict_raw_data.values():
            # the end date should be at the end
            end_num = raw_data.Time_open.iloc[-1]
            end_nums.append(end_num)
        min_end_num = min(end_nums)
        for asset in self.dict_raw_data.keys():
            self.dict_raw_data[asset] = self.dict_raw_data[asset][self.dict_raw_data[asset].Time_open <= min_end_num]

    def compute_ref_values(self):
        # computes the ref values that are used to normalize the state data
        for asset in self.assets:
            self.dict_spot_ref[asset] = self.dict_raw_data[asset].Close.mean()
            self.dict_volume_ref[asset] = self.dict_raw_data[asset].Volume.mean()
            self.dict_spot_diff_std_ref[asset] = self.dict_raw_data[asset].Close.diff().fillna(0).std()

    def initialize_i_and_state_parameters(self):
        raw_data_size_list = []
        for raw_data in self.dict_raw_data.values():
            raw_data_size_list.append(len(raw_data))

        # the minimum state data size is taken by default
        state_data_size_list = []
        for raw_data_size in raw_data_size_list:
            state_data_size_list.append(int(raw_data_size*self.state_data_proportion))
        self.state_data_size = min(state_data_size_list)
        # the number of steps is the remaining steps for the data with the minimum size
        self.N_steps = min(raw_data_size_list) - self.state_data_size + 1

        # then i is set for each asset
        for i in range(len(raw_data_size_list)):
            # set all the start/end i for each asset
            self.i_series_start.iloc[i] = raw_data_size_list[i] - self.N_steps + 1 - self.state_data_size
            self.i_series_end.iloc[i] = raw_data_size_list[i] - self.N_steps + 1

    def set_state_data(self):
        for asset, raw_data in self.dict_raw_data.items():
            self.dict_state_raw_data[asset] = raw_data.iloc[self.i_series_start[asset]:self.i_series_end[asset]]

    # --- COMPUTATION FUNCTIONS ---

    @staticmethod
    def get_data_cut_alpha2(df: pd.DataFrame, cut_degree: int) -> pd.DataFrame:
        # return last proportion at degree cut_degree
        # ex: if len = N, then return the int(N/2^N) lines and the last line if N is too big
        N = len(df) - 1
        i_cut = N - N//2**cut_degree
        return df.iloc[i_cut:]

    def get_alpha_2_state_data(self, raw_state_data: pd.DataFrame) -> list:
        # processing interpolations
        alpha2_data = []

        for cut_degree in range(1, self.nb_cut_alpha2 + 1):
            # here close is chosen for prices
            prices_cut = self.get_data_cut_alpha2(raw_state_data, cut_degree)["Close"]

            # data normalisation
            n = len(prices_cut)
            x = np.arange(n) / n
            # the first price should never be exactly 0
            y = prices_cut / prices_cut.iloc[0]

            # only the 2 first coefficients are kept (the 3rd one is approximately 1)
            interp_data = np.polyfit(x, y, 2)[0:2]
            alpha2_data += list(interp_data)

        return alpha2_data

    # --- STATE FUNCTIONS ---

    def get_spot_prices(self) -> dict:
        # returns the spot prices, they are in particular used for the AssetManager class
        spot_list = dict()
        for asset, raw_state_data in self.dict_state_raw_data.items():
            spot_list[asset] = raw_state_data.Close.iloc[-1]
        return spot_list

    def get_market_state_data(self) -> list:
        # returns the current market state data
        #logger.set_start()

        spot_list = []
        ind_list = []
        alpha2_list = []

        for asset in self.assets:
            raw_state_data = self.dict_state_raw_data[asset]
            # spot
            spot_list.append(raw_state_data.Close.iloc[-1] / self.dict_spot_ref[asset])
            # indicators
            indicators = raw_state_data[self.indicator_columns].iloc[-1].copy()
            indicators.MACD = indicators.MACD / self.dict_spot_ref[asset]
            indicators.RSI = indicators.RSI / 100
            indicators.EOM = indicators.EOM * self.dict_volume_ref[asset] / (self.dict_spot_diff_std_ref[asset]**2 *
                                                                             100000000)
            indicators.BBANDS_RANGE = indicators.BBANDS_RANGE / 100
            ind_list += list(indicators)
            # alpha2
            alpha2_list += self.get_alpha_2_state_data(raw_state_data)

        #logger.info("Market state data computed", with_time=True)
        return spot_list + ind_list + alpha2_list

    def next_step(self):
        # computes the next step
        self.i_series_start += 1
        self.i_series_end += 1
        self.step += 1
        self.set_state_data()

    def is_next_step_possible(self) -> bool:
        # returns if next step is possible
        return self.step < self.N_steps

    def get_state_data_dim(self) -> int:
        # returns the length of the list containing the state data
        N = len(self.assets)
        return N + N * 5 + N * self.nb_cut_alpha2 * 2
