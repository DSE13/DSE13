import pandas as pd
import numpy as np
import os # Make sure os is imported

# Load your final processed CSV
# Ensure this path is correct and the file exists
csv_file_path = 'C:/Users/tomsp/OpenVSP2/OpenVSP-3.43.0-win64/xzylo_mirror.csv'
if not os.path.exists(csv_file_path):
    print(f"ERROR: Main CSV file not found at {csv_file_path}")
    exit()
df = pd.read_csv(csv_file_path, sep=';')

# --- Select Data for a Specific Mach Number (if applicable) ---
# For now, let's assume we use all Mach numbers or that Mach is constant.
# If Mach varies and you need Mach-specific tables, this part needs careful handling.
# One common approach is to select the Mach number closest to your design point.
unique_mach_values = df['Mach'].unique()
if len(unique_mach_values) > 1:
    print(f"Warning: Multiple Mach numbers found: {unique_mach_values}.")
    # For simplicity, let's pick the first Mach number or a common one.
    # You might need to iterate or choose a specific Mach.
    target_mach = unique_mach_values[0] # Or a specific value like 0.0
    print(f"Using data for Mach = {target_mach} for 2D LUTS.")
    df_for_lut = df[np.isclose(df['Mach'], target_mach)].copy()
else:
    target_mach = unique_mach_values[0] if len(unique_mach_values) == 1 else 0.0
    print(f"Using data for Mach = {target_mach} (or default if no Mach column/single value).")
    df_for_lut = df.copy()

if df_for_lut.empty:
    print(f"ERROR: No data found for the selected Mach number ({target_mach}). Cannot create LUTS.")
    exit()

# --- Define the coefficients you want to create tables for ---
coefficients_to_map = ['CL', 'CDtot', 'CMy', 'CFx', 'CFy', 'CFz', 'CMx', 'CMz']
aoa_col = 'AoA'
beta_col = 'Beta'

# --- Get unique, sorted breakpoint vectors ---
# Ensure AoA and Beta columns are numeric before creating breakpoints
df_for_lut[aoa_col] = pd.to_numeric(df_for_lut[aoa_col], errors='coerce')
df_for_lut[beta_col] = pd.to_numeric(df_for_lut[beta_col], errors='coerce')
df_for_lut.dropna(subset=[aoa_col, beta_col], inplace=True) # Remove rows where AoA/Beta are NaN

if df_for_lut.empty:
    print(f"ERROR: No valid AoA/Beta data after NaN drop for Mach {target_mach}. Cannot create LUTS.")
    exit()

aoa_breakpoints = np.sort(df_for_lut[aoa_col].unique())
beta_breakpoints = np.sort(df_for_lut[beta_col].unique())

# Check if breakpoints are valid
if len(aoa_breakpoints) < 2 or len(beta_breakpoints) < 2:
    print("ERROR: Not enough unique AoA or Beta breakpoints to form a 2D table.")
    print(f"AoA unique count: {len(aoa_breakpoints)}, Beta unique count: {len(beta_breakpoints)}")
    exit()

print("AoA Breakpoints:", aoa_breakpoints)
print("Beta Breakpoints:", beta_breakpoints)

# --- Create and save each lookup table ---
output_dir = r"C:\Users\tomsp\OneDrive\Documenten\MATLAB"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store table data for .mat file
mat_data_tables = {}

for coeff in coefficients_to_map:
    print(f"\nProcessing coefficient: {coeff}")
    
    if coeff not in df_for_lut.columns:
        print(f"  Coefficient {coeff} not found in DataFrame. Skipping.")
        continue

    # Ensure coefficient column is numeric
    df_for_lut[coeff] = pd.to_numeric(df_for_lut[coeff], errors='coerce')

    # Create a complete grid of (Beta, AoA) using MultiIndex
    # This helps in identifying exactly which points are missing from the original data
    idx = pd.MultiIndex.from_product([beta_breakpoints, aoa_breakpoints], names=[beta_col, aoa_col])
    
    # Select only relevant columns and set index for easier lookup
    # Drop duplicates for (Beta, AoA) combination, keeping the first.
    # This assumes if duplicates exist, the first one is representative or they are identical for coeff.
    df_coeff_data = df_for_lut[[beta_col, aoa_col, coeff]].drop_duplicates(subset=[beta_col, aoa_col], keep='first')
    
    # Pivot the data
    try:
        table_data_df = df_coeff_data.pivot(index=beta_col, columns=aoa_col, values=coeff)
    except ValueError as ve: # Handles cases where duplicates still cause issues after drop_duplicates (should be rare)
        print(f"  Pivot failed for {coeff} even after drop_duplicates: {ve}. Attempting pivot_table with mean.")
        try:
            table_data_df = pd.pivot_table(df_for_lut, index=beta_col, columns=aoa_col, values=coeff, aggfunc='mean')
        except Exception as pve:
            print(f"  pivot_table also failed for {coeff}: {pve}. Skipping this coefficient.")
            continue
            
    # Reindex to ensure the table matches all breakpoints and fill missing spots with NaN
    table_data_df = table_data_df.reindex(index=beta_breakpoints, columns=aoa_breakpoints)

    # --- Advanced NaN Filling using scipy.interpolate.griddata if available ---
    # This is generally more robust for scattered NaNs than simple pandas interpolate/ffill
    table_data_np = table_data_df.to_numpy() # Get current state

    if np.isnan(table_data_np).any():
        print(f"  NaNs found in raw pivoted table for {coeff}. Shape: {table_data_np.shape}")
        try:
            from scipy.interpolate import griddata

            # Create a mask of valid (non-NaN) points
            valid_mask = ~np.isnan(table_data_np)
            
            # Get coordinates of valid points and their values
            points_known = np.array(np.where(valid_mask)).T # (row_idx, col_idx) of valid points
            values_known = table_data_np[valid_mask]
            
            if len(values_known) < 3 : # griddata needs at least 3-4 points for 2D linear/cubic
                 print(f"  Not enough known data points ({len(values_known)}) for {coeff} to use griddata. Using simpler fill.")
                 # Fallback to pandas ffill/bfill if griddata is not feasible
                 filled_df = table_data_df.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
                 filled_df = filled_df.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
                 # Final fallback for any completely NaN rows/cols that couldn't be filled
                 table_data_np = filled_df.fillna(0).to_numpy() 
            else:
                # Get coordinates of NaN points where we want to interpolate
                points_to_interpolate = np.array(np.where(np.isnan(table_data_np))).T
                
                # Interpolate
                interpolated_values = griddata(points_known, values_known, points_to_interpolate, method='linear') # 'linear' or 'cubic' or 'nearest'

                # Fill the NaNs in the original array
                for i, (r_idx, c_idx) in enumerate(points_to_interpolate):
                    if not np.isnan(interpolated_values[i]):
                        table_data_np[r_idx, c_idx] = interpolated_values[i]
                
                print(f"  Filled NaNs using griddata (linear) for {coeff}.")

                # For points outside the convex hull of known data, griddata (linear/cubic) will return NaN.
                # Fill these remaining NaNs (extrapolation regions) using 'nearest' with griddata or pandas ffill/bfill.
                if np.isnan(table_data_np).any():
                    print(f"  NaNs remain after linear griddata for {coeff} (likely extrapolation). Trying 'nearest' or ffill/bfill.")
                    # Option 1: griddata with 'nearest' for extrapolation
                    nan_mask_after_linear = np.isnan(table_data_np)
                    points_to_extrapolate = np.array(np.where(nan_mask_after_linear)).T
                    if points_to_extrapolate.size > 0: # Check if there are points to extrapolate
                        extrapolated_values_nearest = griddata(points_known, values_known, points_to_extrapolate, method='nearest')
                        for i, (r_idx, c_idx) in enumerate(points_to_extrapolate):
                             if not np.isnan(extrapolated_values_nearest[i]):
                                table_data_np[r_idx, c_idx] = extrapolated_values_nearest[i]
                    
                    # Option 2 (Simpler fallback if 'nearest' still leaves NaNs or if preferred): Pandas ffill/bfill
                    if np.isnan(table_data_np).any():
                        temp_df = pd.DataFrame(table_data_np)
                        filled_df = temp_df.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
                        filled_df = filled_df.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
                        table_data_np = filled_df.fillna(0).to_numpy() # Final fill with 0

        except ImportError:
            print("  scipy.interpolate.griddata not found. Using pandas ffill/bfill for NaNs.")
            filled_df = table_data_df.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
            filled_df = filled_df.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
            table_data_np = filled_df.fillna(0).to_numpy() # Final fallback

    # Check for remaining NaNs
    if np.isnan(table_data_np).any():
        print(f"  WARNING: NaNs are STILL PRESENT in {coeff} table after all filling attempts. Table might be problematic.")
        # As a last resort, fill any remaining NaNs with 0, but this is a sign of deeper issues.
        table_data_np = np.nan_to_num(table_data_np, nan=0.0)


    # Save the table data for .mat file
    mat_data_tables[f'{coeff}_table'] = table_data_np

    # Save as separate CSVs
    pd.DataFrame(table_data_np).to_csv(os.path.join(output_dir, f'{coeff}_table.csv'), header=False, index=False)
    
    print(f"  Table shape for {coeff}: {table_data_np.shape} (Beta rows, AoA columns)")
    print(f"  Saved {coeff}_table.csv")

# Save breakpoints once
pd.Series(aoa_breakpoints).to_csv(os.path.join(output_dir, 'aoa_breakpoints.csv'), header=False, index=False)
pd.Series(beta_breakpoints).to_csv(os.path.join(output_dir, 'beta_breakpoints.csv'), header=False, index=False)
print("\nSaved aoa_breakpoints.csv and beta_breakpoints.csv")

# Save as a .mat file
try:
    from scipy.io import savemat
    mat_file_content = {
        'aoa_bp': aoa_breakpoints,
        'beta_bp': beta_breakpoints,
        **mat_data_tables # Unpack all stored tables
    }
    savemat(os.path.join(output_dir, "aero_luts_revised.mat"), mat_file_content)
    print("\nSaved all tables and breakpoints to aero_luts_revised.mat")
except ImportError:
    print("\nscipy.io.savemat not found. Skipping .mat file creation. CSV files were created.")
except Exception as e:
    print(f"\nError during .mat file creation: {e}")
    print("Skipping .mat file creation. CSV files might have been created.")

print(f"All LUT data saved in directory: {output_dir}")