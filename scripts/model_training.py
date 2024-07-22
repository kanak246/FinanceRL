# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import STOCK_TICKERS

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(ticker):
    input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    df = pd.read_csv(os.path.join(input_dir, f'{ticker}_features.csv'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close', 'MA_50']].values)
    
    X, y = [], []
    for i in range(60, len(df)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=50, batch_size=32)
    model.save(os.path.join(model_dir, f'{ticker}_lstm.h5'))

if __name__ == "__main__":
    for ticker in STOCK_TICKERS:
        train_lstm_model(ticker)
