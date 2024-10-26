import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def predict_for_next_day(df, target_date, is_Treasury=False, model=None):
    """
    Predict volume-weighted yield for the next day for each unique bond,
    and determine if the yield will go up or down. For bonds with no prior data, output "no data prior to target date".
    Also calculates and prints the Mean Squared Error (MSE) between:
    1. Previous year's yield vs actual yield
    2. Predicted yield vs actual yield
    """
    input_dir = "/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/processed"
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Load the trained model and scalers if no model is provided
    if model is None:
        if is_Treasury:
            model = joblib.load(os.path.join(model_dir, 'sgd_treasury_yield_model.pkl'))
            X_scaler = joblib.load(os.path.join(model_dir, 'X_scaler_treasury.pkl'))
            y_scaler = joblib.load(os.path.join(model_dir, 'y_scaler_treasury.pkl'))
        else:
            model = joblib.load(os.path.join(model_dir, 'sgd_bond_yield_model.pkl'))
            X_scaler = joblib.load(os.path.join(model_dir, 'X_scaler.pkl'))
            y_scaler = joblib.load(os.path.join(model_dir, 'y_scaler.pkl'))
    else:
        # If model is passed, assume scalers are also passed with it, or you can pass them explicitly as well
        X_scaler = joblib.load(os.path.join(model_dir, 'X_scaler.pkl'))  # Adapt based on your scaler location
        y_scaler = joblib.load(os.path.join(model_dir, 'y_scaler.pkl'))

    # Prepare data for prediction
    if is_Treasury:
        prediction_df = df[df['Date'] < target_date].copy()  # Treasury uses 'Date'
    else:
        prediction_df = df[df['trd_exctn_dt'] < target_date].copy()  # Bond data uses 'trd_exctn_dt'

    # Ensure feature engineering is done
    if 'time_to_maturity' not in prediction_df.columns:
        prediction_df['time_to_maturity'] = (prediction_df['maturity_date'] - prediction_df['trd_exctn_dt']).dt.days

    if not is_Treasury:
        if 'bond_sym_id_encoded' not in prediction_df.columns:
            prediction_df['bond_sym_id_encoded'] = prediction_df['bond_sym_id'].astype('category').cat.codes

    bonds_to_predict = df['bond_sym_id'].unique() if not is_Treasury else df['Maturity'].unique()

    # For each bond, predict the next day's yield
    predictions = []
    actual_yields = []
    predicted_yields = []
    previous_year_yields = []

    for bond in bonds_to_predict:
        bond_data = prediction_df[prediction_df['bond_sym_id'] == bond].copy() if not is_Treasury else prediction_df.copy()

        # Check if there's any data prior to the target date for this bond
        if bond_data.empty:
            predictions.append({
                'bond_sym_id': bond,
                'volume_weighted_yield_predicted': 'no data prior to target date',
                'direction': 'N/A',
                'percentage_change': 'N/A',
                'error': 'N/A'
            })
            continue

        # Prepare the input features
        if is_Treasury:
            X_predict = bond_data[['time_to_maturity']].values  # Only 'time_to_maturity' for Treasury
        else:
            X_predict = bond_data[['bond_sym_id_encoded', 'total_daily_volume', 'time_to_maturity']].values
        
        # Apply the X_scaler
        X_predict_scaled = X_scaler.transform(X_predict)

        # Predict yield for the next day
        predicted_yield_scaled = model.predict(X_predict_scaled[-1].reshape(1, -1))  # Use the last known data for prediction

        # Unscale the prediction to get it back to the original range
        predicted_yield = y_scaler.inverse_transform(predicted_yield_scaled.reshape(-1, 1))[0][0]

        previous_yield = bond_data['volume_weighted_yield'].iloc[-1]

        # Get the actual yield from the target date for comparison
        actual_yield_data = df[(df['trd_exctn_dt'] == target_date) & (df['bond_sym_id'] == bond)]
        if not actual_yield_data.empty:
            actual_yield = actual_yield_data['volume_weighted_yield'].iloc[0]
        else:
            actual_yield = None  # Handle case if actual data is missing

        # Add the previous year's yield if available
        previous_year_yield = bond_data['volume_weighted_yield'].iloc[-2] if len(bond_data) > 1 else None

        # Determine the direction of change (up or down)
        direction = 'up' if predicted_yield > previous_yield else 'down'
        percentage_change = ((predicted_yield - previous_yield) / previous_yield) * 100

        # Calculate the error if actual yield is available
        if actual_yield is not None:
            error = predicted_yield - actual_yield
            actual_yields.append(actual_yield)
        else:
            error = 'N/A'

        predicted_yields.append(predicted_yield)
        
        if previous_year_yield is not None:
            previous_year_yields.append(previous_year_yield)
        else:
            # Handle the case where there's no previous year yield
            previous_year_yields.append(previous_yield)  # Assuming same as current if not available

        # Append to predictions
        predictions.append({
            'bond_sym_id': bond,
            'volume_weighted_yield_actual': actual_yield,
            'previous yield': previous_yield,
            'volume_weighted_yield_predicted': predicted_yield,
            'direction': direction,
            'percentage_change': percentage_change,
            'trd_exctn_dt': target_date,
            'error': error
        })

    # Filter out missing values for MSE calculation
    filtered_actual_yields = [y for y in actual_yields if y is not None]
    filtered_predicted_yields = predicted_yields[:len(filtered_actual_yields)]
    filtered_previous_year_yields = previous_year_yields[:len(filtered_actual_yields)]

    # Calculate Mean Squared Error (MSE)
    if filtered_actual_yields:
        mse_previous_vs_actual = mean_squared_error(filtered_previous_year_yields, filtered_actual_yields)
        mse_predicted_vs_actual = mean_squared_error(filtered_predicted_yields, filtered_actual_yields)

        r_squared = r2_score(filtered_actual_yields, filtered_predicted_yields)
        # Set R-squared to 0 if negative
        r_squared = max(0, min(1, r_squared))  
        print(f"Mean Squared Error (Previous Year vs Actual): {mse_previous_vs_actual}")
        print(f"Mean Squared Error (Predicted vs Actual): {mse_predicted_vs_actual}")
        print(f"R-squared score: {r_squared}")
    else:
        print("No actual yields available for MSE calculation")

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(input_dir, f'predictions_{target_date}.csv'), index=False)
    print(f"Predictions saved for {target_date} in {input_dir}.")

    return predictions_df
