# trading.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from config import STOCK_TICKERS

def execute_trades():
    input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    for ticker in STOCK_TICKERS:
        model = load_model(os.path.join(model_dir, f'{ticker}_lstm.h5'))
        df = pd.read_csv(os.path.join(input_dir, f'{ticker}_features.csv'))
        data = df['Close'].values
        X_test = np.array([data[-60:]])  # Last 60 days data
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        prediction = model.predict(X_test)
        print(f"Prediction for {ticker}: {prediction[0][0]}")

if __name__ == "__main__":
    execute_trades()
