import pandas as pd
from config import STOCK_TICKERS


def clean_stock_data(ticker):
    df = pd.read_csv(f'../data/raw/{ticker}.csv')
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.to_csv(f'../data/processed/{ticker}_cleaned.csv')

if __name__ == "__main__":
    for ticker in STOCK_TICKERS:
        clean_stock_data(ticker)
