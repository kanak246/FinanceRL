import os
import pandas as pd
from scripts import model_training, prediction
from scripts.utils import load_and_preprocess_data, compare_actual_vs_predicted

if __name__ == "__main__":
    # Set the target date for prediction
    target_date = pd.to_datetime("2023-09-12")
    # target_date = pd.to_datetime("2023-09-28")
    is_Treasury = False
    # Load the dataset and preprocess it
    input_file = '/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/raw/all_aggregated_bond_data.csv'
    # input_file = '/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/raw_copy/raw/Cleaned_Treasury_Yields.csv'
    df, training_df = load_and_preprocess_data(input_file, target_date, is_Treasury)

    # Step 1: Train the model on the available data up till one day of the target date 
    print("Training the model...")
    model, X_scaler, y_scaler = model_training.train_sgd_model(training_df, target_date, is_Treasury, is_incremental=True)

    # Step 2: Predict bond prices 
    print(f"Predicting prices for {target_date}...")
    predictions_df = prediction.predict_for_next_day(df, target_date, is_Treasury)

    # Step 3: Compare predictions with actual values for 20
    if predictions_df is not None:
        comparison_df = compare_actual_vs_predicted(df, target_date, predictions_df, is_Treasury)

        if is_Treasury:
            # Save the comparison results
            output_file = os.path.join('//Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed/treasury_comparison.csv')
        else: 
             output_file = os.path.join('//Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed/bond_comparison.csv')
        comparison_df.to_csv(output_file, index=False)
        print(f"Comparison file saved to {output_file}")
