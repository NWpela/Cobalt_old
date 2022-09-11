import pandas as pd
from little_logger import Little_logger
from ta.trend import MACD, ADXIndicator, CCIIndicator, ema_indicator
from ta.momentum import RSIIndicator
from ta.volume import volume_weighted_average_price, ease_of_movement
from ta.volatility import BollingerBands

"""
    This files aims to provide a set of function to compute technical indicators for given datasets
    There is 2 types of functions:
        - One shot functions: The indicator is computed for the whole dataset
        - Recurrent functions: The indicator is computed using its last values and returns the new value
"""

logger = Little_logger("indicators")

# --- ONE SHOT FUNCTIONS ---

# - Trend -
def compute_MACD(df: pd.DataFrame):
    # df is supposed to contain a "Close" column
    macd = MACD(close=df.Close, fillna=True)
    df["MACD"] = macd.macd_diff()

def compute_baselines(ema_win_list: list, df: pd.DataFrame):
    # df is supposed to contain a "Close" column
    for k in ema_win_list:
        ema_name = f"EMA_{k}"
        df[ema_name] = ema_indicator(df.Close, window=k, fillna=True)
    df["EMA_0"] = df.Close
    df["EMA_LAST"] = df.Close.mean()

    # computing baselines
    ema_win_list_extended = [0] + ema_win_list + ["LAST"]
    for k in range(len(ema_win_list_extended) - 1):
        baseline_name = f"BASELINE_{k}"
        ema_name_1 = f"EMA_{ema_win_list_extended[k]}"
        ema_name_2 = f"EMA_{ema_win_list_extended[k + 1]}"
        df[baseline_name] = df[ema_name_1] - df[ema_name_2]

def compute_ADX(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low" and "Close" column
    # /!\ the function currently encounters troubles calculating ADX
    adx = ADXIndicator(high=df.High, low=df.Low, close=df.Close, fillna=True)
    df["ADX"] = adx.adx()


def compute_CCI(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low" and "Close" column
    cci = CCIIndicator(high=df.High, low=df.Low, close=df.Close, fillna=True)
    df["ADX"] = cci.cci()


# - Momentum -
def compute_RSI(df: pd.DataFrame):
    # df is supposed to contain a "Close" column
    rsi = RSIIndicator(close=df.Close, fillna=True)
    df["RSI"] = rsi.rsi()


# - Volume -
def compute_VWAP(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low", "Close" and "Volume" column
    vwap = volume_weighted_average_price(low=df.Low, high=df.High, close=df.Close, volume=df.Volume, fillna=True)
    df["VWAP"] = vwap

def compute_EOM(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low" and "Volume" column
    eom = ease_of_movement(low=df.Low, high=df.High, volume=df.Volume, fillna=True)
    df["EOM"] = eom

# - Volatility -
def compute_BBANDS(df: pd.DataFrame):
    # df is supposed to contain a "Close" column
    bbands = BollingerBands(close=df.Close, fillna=True)
    df["BBANDS_PERCENT"] = bbands.bollinger_pband()
    df["BBANDS_RANGE"] = bbands.bollinger_wband()
