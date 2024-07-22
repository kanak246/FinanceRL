import os
import yfinance as yf
from config import STOCK_TICKERS, START_DATE, END_DATE

def collect_stock_data():
    # Create the directory for raw data if it does not exist
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)  # Use exist_ok=True to prevent the function from throwing an error if the directory already exists

    for ticker in STOCK_TICKERS:
        print(f"Downloading data for {ticker}")
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        file_path = os.path.join(raw_data_dir, f'{ticker}.csv')
        data.to_csv(file_path)
        print(f"Data for {ticker} saved to {file_path}")

if __name__ == "__main__":
    collect_stock_data()
