# data_processing.py

import os
import pandas as pd
from config import STOCK_TICKERS

def process_data():
    input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker in STOCK_TICKERS:
        df = pd.read_csv(os.path.join(input_dir, f'{ticker}.csv'))
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df[['Close', 'MA_50']].to_csv(os.path.join(output_dir, f'{ticker}_features.csv'))

if __name__ == "__main__":
    process_data()
