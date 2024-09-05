import os
import numpy as np
import pandas as pd
import joblib  # For loading RandomForest or similar models

def calculate_time_to_maturity(df):
    # Ensure the dates are in datetime format
    df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])  # Adjust column name if needed
    df['maturity_date'] = pd.to_datetime(df['maturity_date'])  # Adjust column name if needed
    
    # Calculate time to maturity (in days)
    df['time_to_maturity'] = (df['maturity_date'] - df['trd_exctn_dt']).dt.days
    return df

def execute_bond_trades():
    input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    output_dir = "/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed"
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Load the trained bond model
    model = joblib.load(os.path.join(model_dir, 'bond_yield_model.pkl'))
    
    # Load the bond dataset
    df = pd.read_csv(os.path.join(input_dir, 'aggregated_bond_data.csv'))  # Adjust to your actual file
    
    # Calculate time to maturity if it doesn't exist
    if 'time_to_maturity' not in df.columns:
        df = calculate_time_to_maturity(df)
    
    # Ensure bond_sym_id is encoded as it was during training
    if 'bond_sym_id_encoded' not in df.columns:
        df['bond_sym_id_encoded'] = df['bond_sym_id'].astype('category').cat.codes
    
    # Prepare the features for prediction (using all 3 features)
    features = ['bond_sym_id_encoded', 'total_daily_volume', 'time_to_maturity']  # Adjust based on your file
    
    # Prepare the test data (X) from the features
    X_test = df[features].values  # Convert the relevant features into a NumPy array for prediction
    
    # Make predictions for volume-weighted yield for the bonds
    predictions = model.predict(X_test)
    
    # Add predictions to the DataFrame
    df['predicted_volume_weighted_yield'] = predictions  # Create a new column for the predictions
    
    # Output the results to a new CSV file
    output_file = os.path.join(output_dir, 'bond_predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return df

if __name__ == "__main__":
    execute_bond_trades()
