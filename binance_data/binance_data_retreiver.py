from binance.client import Client
import pandas as pd
import datetime as dt

# API keys and client declaration
__API_KEY = r"oU8OO3HyocqzxYy4Vfiljwd6AXLGUCXLBfnOLatBdl6eGbhzVe56NZrrtoqCEBy8"
__API_SECRET = r"jHJfFKhG3B7hJ0TAPozjIVkDX9vNHnGfQjAe5HEUIMoZKI73nceE4bCDVgJHK593"

client = Client(__API_KEY, __API_SECRET)

# Global parameters
COIN_CODES = ["BTC",
              "ETH",
              "XRP",
              "LINK",
              "XLM",
              "EOS",
              "BNB",
              "TRX",
              "DOGE",
              "MATIC",
              "AVAX",
              "SOL",
              "UNI"]
interval = Client.KLINE_INTERVAL_15MINUTE
start_date_str = "1 May, 2021"
column_names = ["Timestamp_open",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close_time",
                "Quote_asset_volume",
                "Number_of_trades",
                "Taker_buy_base_asset_volume",
                "Taker_buy_quote_asset_volume",
                "Ignore"]
version = "v1"

for COIN_CODE in COIN_CODES:
    PAIR = COIN_CODE + "EUR"
    # Start retrieval using API
    raw_data = client.get_historical_klines(PAIR, interval, start_date_str)

    # Formatting the data and converting to CSV
    data_df = pd.DataFrame(data=raw_data, columns=column_names).drop(columns=["Ignore"])
    data_df["Time_open"] = data_df["Timestamp_open"].apply(lambda x: dt.datetime.fromtimestamp(x/1000))
    data_df["Time_open"] = data_df["Time_open"].dt.strftime("%Y%m%d%H%M%S")

    file_name = '_'.join([PAIR, interval, version]) + ".csv"
    data_df.to_csv(file_name, sep=";")
    print(f"Done for {PAIR}: {file_name} created")
