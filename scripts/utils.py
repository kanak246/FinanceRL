import pandas as pd

def load_and_preprocess_data(input_file, target_date):
    """
    Load and preprocess bond data up to the target date.
    """
    df = pd.read_csv(input_file)

    # Convert dates to datetime using 'errors="coerce"' to handle invalid dates
    df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'], errors='coerce')
    df['maturity_date'] = pd.to_datetime(df['maturity_date'], errors='coerce')

    # Drop rows where dates are not valid (i.e., where there are NaT values after conversion)
    df = df.dropna(subset=['trd_exctn_dt', 'maturity_date'])

    # Calculate time to maturity (in days)
    if 'time_to_maturity' not in df.columns:
        df['time_to_maturity'] = (df['maturity_date'] - df['trd_exctn_dt']).dt.days

    # Encode bond symbol ID
    if 'bond_sym_id_encoded' not in df.columns:
        df['bond_sym_id_encoded'] = df['bond_sym_id'].astype('category').cat.codes

    # Filter training data up to one day before the target date
    training_df = df[df['trd_exctn_dt'] < target_date].copy()

    return df, training_df


def compare_actual_vs_predicted(df, target_date, predictions_df):
    """
    Compare actual volume-weighted yields with predicted yields for a given target date.
    """
    # Filter the actual data for the target date
    actual_df = df[df['trd_exctn_dt'] == target_date][['bond_sym_id', 'volume_weighted_yield']].copy()
    actual_df.rename(columns={'volume_weighted_yield': 'volume_weighted_yield_actual'}, inplace=True)

    # Merge actual and predicted dataframes
    comparison_df = pd.merge(actual_df, predictions_df, on='bond_sym_id', how='left')

    # Calculate the error only for rows where both actual and predicted yields are available
    comparison_df['error'] = comparison_df.apply(
        lambda row: row['volume_weighted_yield_actual'] - row['volume_weighted_yield_predicted']
        if isinstance(row['volume_weighted_yield_predicted'], (int, float))
        else 'N/A',
        axis=1
    )

    # Save the comparison dataframe to a CSV file
    comparison_df.to_csv('comparison.csv', index=False)

    return comparison_df

