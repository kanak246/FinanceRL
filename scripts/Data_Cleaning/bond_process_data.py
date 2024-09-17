import pandas as pd

# Define paths
master_file_path = "/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/raw/Master_File.csv"
bonds_file_path = "/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/raw/Bonds_Sep_2023.csv"

# Load the CSV files
master_data = pd.read_csv(master_file_path)
bonds_data = pd.read_csv(bonds_file_path)

# Merge the datasets on the common bond identifier column
merged_data = pd.merge(bonds_data, master_data, on='bond_sym_id')

# Filter out specific columns and bonds you want to look at, and add the volume quantity
desired_cols = ["bond_sym_id", "yld_pt", "trd_exctn_dt", "mtrty_dt", "buy_cmsn_rt", "sell_cmsn_rt", "entrd_vol_qt"]
desired_bond_sym_id = ["AFL.GF", "AFL4404390", "AFL4972576"]
desired_trd_exctn_dt = ["2023-09-01", "2023-09-05", "2023-09-06"]

# Apply those specific parameters to the filtered dataset
filtered_data = merged_data[desired_cols]
filtered_data = filtered_data[filtered_data["bond_sym_id"].isin(desired_bond_sym_id)]
filtered_data = filtered_data[filtered_data["trd_exctn_dt"].isin(desired_trd_exctn_dt)]

# Group by bond symbol and trade execution date, and aggregate the data
# Calculate the volume-weighted yield and sum the total daily quantity (entrd_vol_qt)
grouped_data = filtered_data.groupby(['trd_exctn_dt', 'bond_sym_id']).agg(
    volume_weighted_yield=('yld_pt', lambda x: (x * filtered_data.loc[x.index, 'entrd_vol_qt']).sum() / filtered_data.loc[x.index, 'entrd_vol_qt'].sum()),
    total_daily_volume=('entrd_vol_qt', 'sum'),
    total_buy_commission=('buy_cmsn_rt', 'sum'),  # Sum up the buy commissions
    total_sell_commission=('sell_cmsn_rt', 'sum'),  # Sum up the sell commissions
    maturity_date=('mtrty_dt', 'first')  # Use the first maturity date (assuming it's the same for all rows)
).reset_index()

# If you want to ignore commissions (since they're mostly zero), you can drop those columns
grouped_data = grouped_data.drop(columns=['total_buy_commission', 'total_sell_commission'])

# Save the grouped dataset to a CSV file
output_path = '/Users/kanakgarg/Desktop/Kanak/FinanceRL/data/raw/aggregated_bond_data.csv'
grouped_data.to_csv(output_path, index=False)

# Display the first few rows of the resulting dataset to confirm
print(grouped_data.head())
