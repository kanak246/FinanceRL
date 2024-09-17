import os
import joblib
import pandas as pd
from scripts.model_setup import create_model
from sklearn.preprocessing import StandardScaler

def train_sgd_model(training_df, target_date, is_incremental=False):
    """
    Train the SGDRegressor model either from scratch or incrementally.
    """
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_file = os.path.join(model_dir, 'sgd_bond_yield_model.pkl')
    X_scaler_file = os.path.join(model_dir, 'X_scaler.pkl')
    y_scaler_file = os.path.join(model_dir, 'y_scaler.pkl')

    # Feature and target extraction
    X_train = training_df[['bond_sym_id_encoded', 'total_daily_volume', 'time_to_maturity']]
    y_train = training_df['volume_weighted_yield'].values.reshape(-1, 1)  # Reshape for scaling

    # Initialize or load the existing model
    if is_incremental and os.path.exists(model_file) and os.path.exists(X_scaler_file) and os.path.exists(y_scaler_file):
        model = joblib.load(model_file)
        X_scaler = joblib.load(X_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        print("Loaded existing model and scalers for incremental training...")
    else:
        model = create_model()  #Create Model from model_setup page 
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

    # Scale the features and target
    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    # Save the scalers
    joblib.dump(X_scaler, X_scaler_file)
    joblib.dump(y_scaler, y_scaler_file)

    # Train the model
    model.partial_fit(X_train_scaled, y_train_scaled.ravel())  # Use partial_fit for incremental learning

    # Save the model to disk
    joblib.dump(model, model_file)
    print(f"Model trained and saved to {model_file}.")

    return model, X_scaler, y_scaler
