import quantstats as qs
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_performance(actual_yields, predicted_yields, output_file='report.html'):
    """
    Generate an HTML report evaluating bond yield predictions.
    Includes error metrics and visual performance analysis.
    """
    # Calculate error metrics
    mse = mean_squared_error(actual_yields, predicted_yields)
    mae = mean_absolute_error(actual_yields, predicted_yields)
    r2 = r2_score(actual_yields, predicted_yields)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    
    # Create a DataFrame to hold actual and predicted yields as if they were returns
    performance_data = pd.DataFrame({
        'actual_yield': actual_yields,
        'predicted_yield': predicted_yields
    }, index=actual_yields.index)

    # QuantStats requires a returns series, so we treat yields as returns for reporting purposes
    qs.reports.html(performance_data['predicted_yield'], 
                    benchmark=performance_data['actual_yield'], 
                    output=output_file)

def generate_sample_yield_data():
    """
    Generates sample actual and predicted yield data for demonstration purposes.
    In practice, this data will come from the model predictions.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-01-01')
    
    # Sample actual yields
    actual_yields = np.random.normal(5, 0.5, len(dates))  # Mean yield of 5%, stddev of 0.5%
    
    # Sample predicted yields (slightly noisy versions of actual yields)
    predicted_yields = actual_yields + np.random.normal(0, 0.1, len(dates))  # Add some noise to simulate predictions
    
    actual_yields_series = pd.Series(actual_yields, index=dates)
    predicted_yields_series = pd.Series(predicted_yields, index=dates)
    
    return actual_yields_series, predicted_yields_series

if __name__ == "__main__":
    # Generate sample actual and predicted yield data
    actual_yields, predicted_yields = generate_sample_yield_data()

    # Evaluate the performance of the yield predictions and generate an HTML report
    evaluate_performance(actual_yields, predicted_yields)
