import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scripts import model_training, prediction
from scripts.utils import load_and_preprocess_data, compare_actual_vs_predicted

def plot_accuracy_vs_percentage_change(predictions_df, model_type):
    """
    Plots the accuracy of correct predictions (direction) across different percentage change bins.
    """
    # Ensure percentage_change is numeric
    predictions_df['percentage_change'] = pd.to_numeric(predictions_df['percentage_change'], errors='coerce')
    
    # Drop rows with NaN in percentage_change
    predictions_df = predictions_df.dropna(subset=['percentage_change'])

    # Ensure error column is numeric
    predictions_df['error'] = pd.to_numeric(predictions_df['error'], errors='coerce')
    
    # Drop rows where error is NaN
    predictions_df = predictions_df.dropna(subset=['error'])

    # Define bins for percentage change
    bins = [-100, -10, -5, -2, 0, 2, 5, 10, 100]  
    labels = ['<-10%', '-10% to -5%', '-5% to -2%', '-2% to 0%', '0% to 2%', '2% to 5%', '5% to 10%', '>10%']

    # Assign percentage bins
    predictions_df['percentage_bin'] = pd.cut(predictions_df['percentage_change'], bins=bins, labels=labels)

    # Calculate accuracy per bin
    predictions_df['is_correct'] = (
        ((predictions_df['direction'] == 'up') & (predictions_df['error'] > 0)) |
        ((predictions_df['direction'] == 'down') & (predictions_df['error'] < 0))
    )
    accuracy_per_bin = predictions_df.groupby('percentage_bin')['is_correct'].mean()

    # Plotting accuracy vs. percentage change
    plt.figure(figsize=(10, 6))
    accuracy_per_bin.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Percentage Change Bins')
    plt.ylabel('Fraction of Correct Predictions')
    plt.title(f'Accuracy vs. Percentage Change Bins for {model_type}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_data_distribution(predictions_df, model_type):
    """
    Plots the number of data entries in each percentage change bin.
    """
    # Ensure percentage_change is numeric
    predictions_df['percentage_change'] = pd.to_numeric(predictions_df['percentage_change'], errors='coerce')
    
    # Drop rows with NaN in percentage_change
    predictions_df = predictions_df.dropna(subset=['percentage_change'])

    # Define bins for percentage change
    bins = [-100, -10, -5, -2, 0, 2, 5, 10, 100]  
    labels = ['<-10%', '-10% to -5%', '-5% to -2%', '-2% to 0%', '0% to 2%', '2% to 5%', '5% to 10%', '>10%']

    # Assign percentage bins
    predictions_df['percentage_bin'] = pd.cut(predictions_df['percentage_change'], bins=bins, labels=labels)

    # Count the number of data points in each bin
    bin_counts = predictions_df['percentage_bin'].value_counts(sort=False) 

    # Plotting data distribution across bins
    plt.figure(figsize=(10, 6))
    bin_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.xlabel('Percentage Change Bins')
    plt.ylabel('Number of Data Entries')
    plt.title(f'Data Distribution Across Percentage Change Bins for {model_type}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set the target date for prediction
    target_date = pd.to_datetime("2023-09-12")
    is_Treasury = False

    # Load the dataset and preprocess it
    input_file = '/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/raw/all_aggregated_bond_data.csv'
    df, training_df = load_and_preprocess_data(input_file, target_date, is_Treasury)

    # List of all available model types
    model_list = ['sgd', 'random_forest', 'decision_tree', 'xgboost', 'lasso', 'ridge', 'elasticnet', 'svr', 'knn', 'lightgbm', 'bayesian_ridge']

    # Choose the model type you want to train
    model_type = 'bayesian_ridge'  # Change this to any model you want from model_list

    # Extract features and target for cross-validation and training
    if is_Treasury:
        X_train = training_df[['time_to_maturity']]
    else:
        X_train = training_df[['bond_sym_id_encoded', 'total_daily_volume', 'time_to_maturity']]
    y_train = training_df['volume_weighted_yield'].values

    # Train the model
    print(f"Training the {model_type} model...")
    best_model, X_scaler, y_scaler = model_training.train_model(
        training_df, target_date, model_type, is_Treasury, is_incremental=True, tune_params=False
    )

    # Step 2: Use the best_model for predicting bond prices
    print(f"Predicting prices for {target_date}...")
    predictions_df = prediction.predict_for_next_day(df, target_date, is_Treasury, model=best_model)

    # Step 3: Compare predictions with actual values
    if predictions_df is not None:
        comparison_df = compare_actual_vs_predicted(df, target_date, predictions_df, is_Treasury)

        # Save the comparison results
        output_file = os.path.join(
            '/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed',
            f'bond_comparison_{model_type}.csv'
        )
        comparison_df.to_csv(output_file, index=False)
        print(f"Comparison file saved to {output_file}")

    # Plot accuracy vs. percentage change and data distribution
    if predictions_df is not None:
        plot_accuracy_vs_percentage_change(predictions_df, model_type)
        plot_data_distribution(predictions_df, model_type)

    # Perform cross-validation and print the CV scores
    print(f"Performing cross-validation for {model_type} model...")
    cv_mse, cv_r2 = model_training.cross_validate_model(X_train, y_train, model_type=model_type)
    print(f"Cross-Validation MSE ({model_type}): {cv_mse}")
    print(f"Cross-Validation R-squared ({model_type}): {cv_r2}")
