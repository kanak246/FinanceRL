import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scripts import model_training, prediction
from scripts.utils import load_and_preprocess_data, compare_actual_vs_predicted

def plot_accuracy_vs_percentage_change(predictions_df, model_type):
    """
    Plots the accuracy of correct predictions (direction) across different percentage change bins with thinner ranges and no 0% bins.
    """
    # Ensure percentage_change is numeric
    predictions_df['percentage_change'] = pd.to_numeric(predictions_df['percentage_change'], errors='coerce')
    
    # Drop rows with NaN in percentage_change
    predictions_df = predictions_df.dropna(subset=['percentage_change'])

    # Ensure error column is numeric
    predictions_df['error'] = pd.to_numeric(predictions_df['error'], errors='coerce')
    
    # Drop rows where error is NaN
    predictions_df = predictions_df.dropna(subset=['error'])

    # Define thinner bins for percentage change, excluding 0
    bins = [-100, -10, -5, -3, -1, 1, 3, 5, 10, 100]
    labels = ['-100% to -10%', '-10% to -5%', '-5% to -3%', '-3% to -1%', '-1% to 1%', '1% to 3%', '3% to 5%', '5% to 10%', '10% to 100%']

    # Ensure the number of labels matches the number of bins minus one
    if len(bins) - 1 != len(labels):
        raise ValueError("Number of labels must be one fewer than the number of bin edges.")

    # Assign percentage bins
    predictions_df['percentage_bin'] = pd.cut(predictions_df['percentage_change'], bins=bins, labels=labels)

    # Calculate accuracy per bin
    predictions_df['is_correct'] = (
        ((predictions_df['direction'] == 'up') & (predictions_df['error'] > 0)) |
        ((predictions_df['direction'] == 'down') & (predictions_df['error'] < 0))
    )
    accuracy_per_bin = predictions_df.groupby('percentage_bin')['is_correct'].mean()

    # Plotting accuracy vs. percentage change
    plt.figure(figsize=(12, 6))
    accuracy_per_bin.plot(kind='bar', color='lightgreen', edgecolor='black')
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
    predictions_df['percentage_change'] = pd.to_numeric(predictions_df['percentage_change'], errors='coerce')
    predictions_df = predictions_df.dropna(subset=['percentage_change'])

    bins = [-100, -10, -5, -3, -1, 1, 3, 5, 10, 100]
    labels = ['-100% to -10%', '-10% to -5%', '-5% to -3%', '-3% to -1%', '-1% to 1%', '1% to 3%', '3% to 5%', '5% to 10%', '10% to 100%']

    predictions_df['percentage_bin'] = pd.cut(predictions_df['percentage_change'], bins=bins, labels=labels)

    bin_counts = predictions_df['percentage_bin'].value_counts(sort=False)

    plt.figure(figsize=(10, 6))
    bin_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.xlabel('Percentage Change Bins')
    plt.ylabel('Number of Data Entries')
    plt.title(f'Data Distribution Across Percentage Change Bins for {model_type}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compute_accuracy_and_fraction(predictions_df, interval_steps):
    """
    Computes the accuracy rate for predictions outside the interval [-x%, x%] and the fraction of data points within the interval.
    """
    predictions_df['percentage_change'] = pd.to_numeric(predictions_df['percentage_change'], errors='coerce')
    predictions_df['error'] = pd.to_numeric(predictions_df['error'], errors='coerce')
    predictions_df = predictions_df.dropna(subset=['percentage_change', 'error'])

    predictions_df['is_correct'] = (
        ((predictions_df['direction'] == 'up') & (predictions_df['error'] > 0)) |
        ((predictions_df['direction'] == 'down') & (predictions_df['error'] < 0))
    )

    fractions_within_interval = []
    accuracy_outside_interval = []

    for x in interval_steps:
        within_interval = predictions_df[(predictions_df['percentage_change'] >= -x) & (predictions_df['percentage_change'] <= x)]
        fraction_within = len(within_interval) / len(predictions_df)

        outside_interval = predictions_df[(predictions_df['percentage_change'] < -x) | (predictions_df['percentage_change'] > x)]
        accuracy_outside = outside_interval['is_correct'].mean() if len(outside_interval) > 0 else 0

        fractions_within_interval.append(fraction_within)
        accuracy_outside_interval.append(accuracy_outside)

    return fractions_within_interval, accuracy_outside_interval

def plot_pareto_frontier(fractions_within, accuracies_outside, interval_steps, model_type):
    """
    Plots the Pareto frontier: accuracy outside of the interval [-x%, x%] vs. fraction of data points within [-x%, x%].
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fractions_within, accuracies_outside, marker='o', linestyle='-', color='b')
    for i, x in enumerate(interval_steps):
        plt.annotate(f'{x}%', (fractions_within[i], accuracies_outside[i]))

    plt.xlabel('Fraction of Data Points Within Interval [-x%, x%]')
    plt.ylabel('Accuracy Rate for Entries Outside Interval [-x%, x%]')
    plt.title(f'Pareto Frontier Plot for {model_type}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set the target date for prediction
    target_date = pd.to_datetime("2023-09-12")
    is_Treasury = False

    # Load the dataset and preprocess it
    input_file = '/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/raw/all_aggregated_bond_data.csv'
    df, training_df = load_and_preprocess_data(input_file, target_date, is_Treasury)

    model_list = ['sgd', 'random_forest', 'decision_tree', 'xgboost', 'lasso', 'ridge', 'elasticnet', 'svr', 'knn', 'lightgbm', 'bayesian_ridge']
    model_type = 'sgd'  # Change as needed

    if is_Treasury:
        X_train = training_df[['time_to_maturity']]
    else:
        X_train = training_df[['bond_sym_id_encoded', 'total_daily_volume', 'time_to_maturity']]
    y_train = training_df['volume_weighted_yield'].values

    tune_params = False
    if tune_params:
        print(f"Tuning hyperparameters for {model_type} model...")
        best_model, best_params, best_score = model_training.tune_hyperparameters(X_train, y_train, model_type=model_type)
        print(f"Best parameters for {model_type}: {best_params}")
        print(f"Best CV MSE for {model_type}: {best_score}")
    else:
        print(f"Training the {model_type} model...")
        best_model, X_scaler, y_scaler = model_training.train_model(training_df, target_date, model_type, is_Treasury, is_incremental=True, tune_params=tune_params)

    print(f"Predicting prices for {target_date}...")
    predictions_df = prediction.predict_for_next_day(df, target_date, is_Treasury, model=best_model)

    if predictions_df is not None:
        comparison_df = compare_actual_vs_predicted(df, target_date, predictions_df, is_Treasury)
        output_file = f'/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed/bond_comparison_{model_type}.csv'
        comparison_df.to_csv(output_file, index=False)
        print(f"Comparison file saved to {output_file}")

        #Plot Accuracy 
        plot_accuracy_vs_percentage_change(predictions_df, model_type)
        # Plot data distribution
        plot_data_distribution(predictions_df, model_type)

        # Compute and plot Pareto frontier
        interval_steps = [1, 2, 3, 5, 10]
        fractions_within, accuracies_outside = compute_accuracy_and_fraction(predictions_df, interval_steps)
        plot_pareto_frontier(fractions_within, accuracies_outside, interval_steps, model_type)

    print(f"Performing cross-validation for {model_type} model...")
    cv_mse, cv_r2 = model_training.cross_validate_model(X_train, y_train, model_type=model_type)
    print(f"Cross-Validation MSE ({model_type}): {cv_mse}")
    print(f"Cross-Validation R-squared ({model_type}): {cv_r2}")
