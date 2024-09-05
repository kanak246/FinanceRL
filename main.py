import pandas as pd
from scripts import model_training, trading
from scripts.performance_eval import evaluate_performance

if __name__ == "__main__":
    # Step 1: Train the model on the cleaned bond dataset
    print("Training the bond model...")
    model_training.train_model()

    # Step 2: Execute trades and get the DataFrame with predictions
    print("Executing trades based on the trained model...")
    df_with_predictions = trading.execute_bond_trades()

    # Step 3: Ensure the 'trd_exctn_dt' (or similar) is set as the index for QuantStats
    df_with_predictions['trd_exctn_dt'] = pd.to_datetime(df_with_predictions['trd_exctn_dt'])  # Convert to datetime if necessary
    df_with_predictions.set_index('trd_exctn_dt', inplace=True)  # Set the date column as index

    # Step 4: Check for and handle duplicate dates
    if df_with_predictions.index.duplicated().any():
        print("Duplicate dates found. Aggregating by averaging numeric values.")
        # Aggregate numeric columns only (like volume_weighted_yield, total_daily_volume, etc.)
        df_with_predictions = df_with_predictions.groupby(df_with_predictions.index).mean(numeric_only=True)

    # Step 5: Evaluate performance (using the predicted and actual values)
    actual_yields = df_with_predictions['volume_weighted_yield']  # Assuming this column exists in your dataset
    predicted_yields = df_with_predictions['predicted_volume_weighted_yield']
    
    
    # Step 6: Evaluate performance by comparing predicted yields with actual yields
    print("Evaluating performance...")
    evaluate_performance(actual_yields, predicted_yields, output_file='bond_yield_performance_report.html')

    print("Evaluation complete. Check bond_yield_performance_report.html for details.")
