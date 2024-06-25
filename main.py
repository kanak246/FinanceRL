from scripts.data_collection import collect_stock_data
from scripts.data_cleaning import clean_stock_data
from scripts.model_setup import create_model
from scripts.trading import TradingStrategy, execute_trading_strategy
from scripts.training_backtesting import train_model, backtest_strategy
from scripts.performance_evaluation import evaluate_performance
from config import STOCK_TICKERS, START_DATE, END_DATE

if __name__ == "__main__":
    # Data collection and cleaning
    raw_data = collect_stock_data(STOCK_TICKERS, START_DATE, END_DATE)
    for ticker in STOCK_TICKERS:
        clean_stock_data(ticker)

    # Model setup
    model = create_model(input_shape=(None, 5))

    # Training and backtesting
    # Assuming `train_data`, `train_labels`, `test_data`, `test_labels` are prepared
    train_model(model, train_data, train_labels)
    strategy = TradingStrategy(model)
    accuracy = backtest_strategy(strategy, test_data, test_labels)

    # Performance evaluation
    # Assuming `returns` is prepared
    evaluate_performance(returns)
