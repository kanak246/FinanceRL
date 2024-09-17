import quantstats as qs
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_performance(actual_yields, predicted_yields, output_file='report.html'):
    mse = mean_squared_error(actual_yields, predicted_yields)
    mae = mean_absolute_error(actual_yields, predicted_yields)
    r2 = r2_score(actual_yields, predicted_yields)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    
    # Create a DataFrame to hold actual and predicted yields
    performance_data = pd.DataFrame({
        'actual_yield': actual_yields,
        'predicted_yield': predicted_yields
    }, index=actual_yields.index)

    # QuantStats requires a returns series, so we treat yields as returns for reporting purposes
    qs.reports.html(performance_data['predicted_yield'], 
                    benchmark=performance_data['actual_yield'], 
                    output=output_file)
