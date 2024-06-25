import numpy as np

def simple_moving_average_strategy(prices, window=20):
    sma = prices.rolling(window=window).mean()
    signals = np.where(prices > sma, 1, -1)
    return signals

def buy_and_hold_strategy(prices):
    return np.ones(len(prices))
