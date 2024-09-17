import os
import joblib
import pandas as pd

def predict_for_next_day(df, target_date):
    """
    Predict volume-weighted yield for the next day for each unique bond,
    and determine if the yield will go up or down. For bonds with no prior data, output "no data prior to target date".
    """
    input_dir = "/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed"
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Load the trained model and scalers
    model = joblib.load(os.path.join(model_dir, 'sgd_bond_yield_model.pkl'))
    X_scaler = joblib.load(os.path.join(model_dir, 'X_scaler.pkl'))
    y_scaler = joblib.load(os.path.join(model_dir, 'y_scaler.pkl'))

    # Prepare data for prediction
    prediction_df = df[df['trd_exctn_dt'] < target_date].copy()  # Exclude target date data

    # Ensure feature engineering is done
    if 'time_to_maturity' not in prediction_df.columns:
        prediction_df['time_to_maturity'] = (prediction_df['maturity_date'] - prediction_df['trd_exctn_dt']).dt.days
    if 'bond_sym_id_encoded' not in prediction_df.columns:
        prediction_df['bond_sym_id_encoded'] = prediction_df['bond_sym_id'].astype('category').cat.codes

    bonds_to_predict = df['bond_sym_id'].unique()  # Unique bonds in the entire dataset

    # For each bond, predict the next day's yield
    predictions = []
    for bond in bonds_to_predict:
        bond_data = prediction_df[prediction_df['bond_sym_id'] == bond].copy()
        
        # Check if there's any data prior to the target date for this bond
        if bond_data.empty:
            predictions.append({
                'bond_sym_id': bond,
                'volume_weighted_yield_predicted': 'no data prior to target date',
                'direction': 'N/A',
                'percentage_change': 'N/A'
            })
            continue
        
        # Prepare the input features
        X_predict = bond_data[['bond_sym_id_encoded', 'total_daily_volume', 'time_to_maturity']].values

        # Apply the X_scaler
        X_predict_scaled = X_scaler.transform(X_predict)
        
        # Predict yield for the next day
        predicted_yield_scaled = model.predict(X_predict_scaled[-1].reshape(1, -1))  # Use the last known data for prediction
        
        # Unscale the prediction to get it back to the original range
        predicted_yield = y_scaler.inverse_transform(predicted_yield_scaled.reshape(-1, 1))[0][0]
        
        previous_yield = bond_data['volume_weighted_yield'].iloc[-1]
        
        # Determine the direction of change (up or down)
        direction = 'up' if predicted_yield > previous_yield else 'down'
        percentage_change = ((predicted_yield - previous_yield) / previous_yield) * 100
        
        predictions.append({
            'bond_sym_id': bond,
            'volume_weighted_yield_predicted': predicted_yield,
            'direction': direction,
            'percentage_change': percentage_change  # Add percentage change to output
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df['trd_exctn_dt'] = target_date  # Set the target date for prediction
    predictions_df.to_csv(os.path.join(input_dir, f'predictions_{target_date}.csv'), index=False)
    print(f"Predictions saved for {target_date} in {input_dir}.")

    return predictions_df
