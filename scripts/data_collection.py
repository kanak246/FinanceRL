import pandas as pd
import yfinance as yf
from config import STOCK_TICKERS, START_DATE, END_DATE

def collect_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    stock_data = collect_stock_data(STOCK_TICKERS, START_DATE, END_DATE)
    for ticker, df in stock_data.items():
        df.to_csv(f'../data/raw/{ticker}.csv')
