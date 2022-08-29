import pandas as pd
from little_logger import Little_logger
from ta.trend import MACD, ADXIndicator, CCIIndicator
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
