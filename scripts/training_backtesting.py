def train_model(model, data, labels, epochs=10, batch_size=32):
    model.fit(data, labels, epochs=epochs, batch_size=batch_size)

def backtest_strategy(strategy, data, labels):
    predictions = strategy.predict(data)
    accuracy = np.mean(predictions == labels)
    return accuracy
