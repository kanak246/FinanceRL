import sys
import os
import joblib
# Add the root directory (where config.py is) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scripts.model_setup import create_model  # Your custom model creation function
from config import BOND_DATA_PATH

def train_model():
    # Load bond data
    bond_data_path = BOND_DATA_PATH
    bond_df = pd.read_csv(bond_data_path)

    # Convert dates to datetime format
    bond_df["trd_exctn_dt"] = pd.to_datetime(bond_df["trd_exctn_dt"])
    bond_df["maturity_date"] = pd.to_datetime(bond_df["maturity_date"])

    # Feature Engineering: Calculate time to maturity in days
    bond_df['time_to_maturity'] = (bond_df['maturity_date'] - bond_df['trd_exctn_dt']).dt.days

    # Encode categorical variables (e.g., bond identifiers)
    encoder = LabelEncoder()
    bond_df['bond_sym_id_encoded'] = encoder.fit_transform(bond_df['bond_sym_id'])

    # Select features and target variable (assuming volume_weighted_yield is the target)
    X = bond_df[["bond_sym_id_encoded", "total_daily_volume", "time_to_maturity"]]
    y = bond_df["volume_weighted_yield"]  # Ensure this is the correct target column

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestRegressor model
    model = create_model()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model trained. MSE: {mse}, R-squared: {r2}")

    # Save the trained model to the models directory
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)  # Create the 'models' directory if it doesn't exist
    joblib.dump(model, os.path.join(model_dir, 'bond_yield_model.pkl'))  # Save the model

if __name__ == "__main__":
    train_model()
