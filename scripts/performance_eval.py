import quantstats as qs
import pandas as pd
import numpy as np

def evaluate_performance(returns):
    # Generate an HTML report for the given returns
    qs.reports.html(returns, output='report.html')

if __name__ == "__main__":
    # Generating a sample returns series for demonstration purposes
    # In practice, you would use actual historical return data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-01-01')
    sample_returns = np.random.normal(0, 0.01, len(dates))
    returns = pd.Series(sample_returns, index=dates)

    # Evaluate the performance of the sample returns
    evaluate_performance(returns)
