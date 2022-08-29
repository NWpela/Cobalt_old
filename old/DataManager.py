import pandas as pd
import math as m
import numpy as np


class DataManager:
    """
        Based on a file and load it to get and the required data
        batch the data to return good parameters for Kobalt (data for episod)
        Currently based on Coinmetrics.io: https://charts.coinmetrics.io/network-data/
    """

    def __init__(self):
        self.state_data_proportion = 1/300
        self.state_data_size = None

        self.next_available = False
        self.raw_df = None

        self.i = 0
        # we first choose 5 cuts, it should be enough
        self.nb_cut_model = 5

    def load_data(self, file: str):
        # file: csv file from data folder
        self.raw_df = pd.read_csv(file)
        self.raw_df.rename({"BTC / USD Denominated Closing Price": "Price"}, inplace=True)
        mask_0 = self.raw_df == 0
        self.raw_df.loc[mask_0, "Price"] = m.log(self.raw_df.loc[mask_0, "Price"])

        self.state_data_size = int(self.state_data_proportion * len(self.raw_df))
        # initialize i
        self.i = 0

    @staticmethod
    def get_data_proportion(df, n_cut):
        # return last proportion at degree n
        # ex: if len = N, then return the int(N/2^N) lines and the last line if N is too big
        N = len(df)
        i_cut = N - N//2**n_cut
        return df.iloc[i_cut:]

    def get_alpha_2_df_last_value(self):
        alpha_2_data = pd.DataFrame()
        state_data = self.raw_df["Price"][self.i:self.i+self.state_data_size]
        # processing interpolations
        for n_cut in range(1, self.nb_cut_model):
            df_cut = self.get_data_proportion(state_data, n_cut)
            interp_data = pd.Series(np.polyfit(np.arange(len(df_cut), state_data["Price"]), 2), index=["a0", "a1", "a2"])
            alpha_2_data.append(interp_data, ignore_index=True)
        # add the last value
        last_value = state_data.Price.iloc[-1]
        return alpha_2_data, last_value

