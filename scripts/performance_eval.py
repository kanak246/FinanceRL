# performance_eval.py

import quantstats as qs
import pandas as pd
import numpy as np

def evaluate_performance(returns, output_file='report.html'):
    # Generate an HTML report for the given returns
    qs.reports.html(returns, output=output_file)

def generate_sample_returns():
    # Generating a sample returns series for demonstration purposes
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-01-01')
    sample_returns = np.random.normal(0, 0.01, len(dates))
    returns = pd.Series(sample_returns, index=dates)
    return returns

if __name__ == "__main__":
    returns = generate_sample_returns()
    evaluate_performance(returns)
