import numpy as np

class TradingStrategy:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return self.model.predict(data)

    def adjust_positions(self, signals):
        positions = np.cumsum(signals)
        return positions

def execute_trading_strategy(strategy, data):
    signals = strategy.predict(data)
    positions = strategy.adjust_positions(signals)
    return positions
