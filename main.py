import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scripts import model_training, prediction
from scripts.utils import load_and_preprocess_data, compare_actual_vs_predicted


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
        predictions_df.to_csv(f'/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed/prediction_{model_type}.csv', index=False)
        print(f"Comparison  and prediction file saved")




    print(f"Performing cross-validation for {model_type} model...")
    cv_mse, cv_r2 = model_training.cross_validate_model(X_train, y_train, model_type=model_type)
    print(f"Cross-Validation MSE ({model_type}): {cv_mse}")
    print(f"Cross-Validation R-squared ({model_type}): {cv_r2}")
