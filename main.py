# main.py

from scripts import data_collection, data_processing, model_training, trading
from scripts.performance_eval import evaluate_performance, generate_sample_returns
from config import STOCK_TICKERS, START_DATE, END_DATE

if __name__ == "__main__":
    data_collection.collect_stock_data()
    data_processing.process_data()
    for ticker in STOCK_TICKERS:
        model_training.train_lstm_model(ticker)
    trading.execute_trades()

    sample_returns = generate_sample_returns()
    evaluate_performance(sample_returns, output_file='trading_performance_report.html')
