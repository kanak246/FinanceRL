import pandas as pd

def maturity_in_days(maturity_str):
    """
    Convert the maturity string (e.g., "1 Mo", "3 Yr") to the number of days.
    """
    if pd.isna(maturity_str):
        return None
    
    if 'Mo' in maturity_str:
        months = int(maturity_str.split()[0])
        return months * 30
    elif 'Yr' in maturity_str:
        years = int(maturity_str.split()[0])
        return years * 365
    else: 
        return 0

def load_and_preprocess_data(input_file, target_date, is_Treasury=False):
    """
    Load and preprocess bond or Treasury data up to the target date.
    """
    df = pd.read_csv(input_file)

    if not is_Treasury:
        print("Processing bond data")
        # Convert dates to datetime using 'errors="coerce"' to handle invalid dates
        df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'], errors='coerce')
        df['maturity_date'] = pd.to_datetime(df['maturity_date'], errors='coerce')

        # Drop rows where dates are not valid (i.e., where there are NaT values after conversion)
        df = df.dropna(subset=['trd_exctn_dt', 'maturity_date'])

        # Calculate time to maturity (in days)
        if 'time_to_maturity' not in df.columns:
            df['time_to_maturity'] = (df['maturity_date'] - df['trd_exctn_dt']).dt.days

        # Create volume_weighted_yield for bonds (assuming you have the necessary columns)
        if 'volume_weighted_yield' not in df.columns:
            df['volume_weighted_yield'] = df['yld_pt']  # Assuming yield is in yld_pt column; adjust if needed

        # Encode bond symbol ID
        if 'bond_sym_id_encoded' not in df.columns:
            df['bond_sym_id_encoded'] = df['bond_sym_id'].astype('category').cat.codes

        # Filter training data up to one day before the target date
        training_df = df[df['trd_exctn_dt'] < target_date].copy()

    else:
        print("Processing Treasury Data")
        # Process Treasury data differently
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Ensure the yield column exists and is labeled as volume_weighted_yield
        if 'volume_weighted_yield' not in df.columns:
            df['volume_weighted_yield'] = df['Yield']  # Assuming Yield column exists

        # Convert Maturity to time to maturity in days (assuming "1 Mo", "3 Mo" etc.)
        if 'time_to_maturity' not in df.columns:
            df['time_to_maturity'] = df['Maturity'].apply(lambda x: maturity_in_days(x))

        # Filter training data up to one day before the target date
        training_df = df[df['Date'] < target_date].copy()

    return df, training_df

def compare_actual_vs_predicted(df, target_date, predictions_df, is_Treasury=False):
    """
    Compare actual volume-weighted yields with predicted yields for a given target date.
    Handles both Treasury and bond datasets.
    """
    # Set the appropriate date column depending on whether it's Treasury or bond data
    date_column = 'Date' if is_Treasury else 'trd_exctn_dt'

    # Check if the appropriate date column exists in the dataframe
    if date_column not in df.columns:
        raise KeyError(f"Expected column '{date_column}' not found in the dataset.")

    # For Treasury data, there may not be a 'bond_sym_id', so we may use a different unique identifier
    bond_id_column = 'bond_sym_id' if 'bond_sym_id' in df.columns else 'Maturity'  # Modify as per your actual Treasury column

    # Filter the actual data for the target date
    actual_df = df[df[date_column] == target_date][[bond_id_column, 'volume_weighted_yield']].copy()
    actual_df.rename(columns={'volume_weighted_yield': 'volume_weighted_yield_actual'}, inplace=True)

    # Merge the actual and predicted dataframes based on bond or Treasury identifier
    comparison_df = pd.merge(actual_df, predictions_df, left_on=bond_id_column, right_on='bond_sym_id', how='left')

    # Calculate the error between actual and predicted yields
    comparison_df['error'] = comparison_df.apply(
        lambda row: row['volume_weighted_yield_actual'] - row['volume_weighted_yield_predicted']
        if isinstance(row['volume_weighted_yield_predicted'], (int, float))
        else 'N/A',
        axis=1
    )

    return comparison_df

