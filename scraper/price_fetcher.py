import yfinance as yf
import pandas as pd

btc = yf.download("BTC-USD", period="60d", interval="1d")

btc.to_csv("../data/raw/btc_price.csv")

print("BTC price data saved")