import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import StringIO

# --- USER: Define Column Names for Data Files ---
# These are the *actual* column names as they appear in your data files.

# For Experimental CSV
EXP_AOA_COL_ACTUAL = 'AoA_Exp'  # Actual AoA column name in your experimental CSV
EXP_CL_COL_ACTUAL = 'CL_Exp'    # Actual CL column name
EXP_CD_COL_ACTUAL = 'CD_Exp'    # Actual CD column name
# Fallback indices if headers are not found (0 for AoA, 1 for CL, 2 for CD)
EXP_AOA_COL_IDX = 0
EXP_CL_COL_IDX = 2
EXP_CD_COL_IDX = 1

# For VSPAERO .polar file (Panel Method)
VSP_AOA_COL_ACTUAL = 'AoA'    # Actual AoA column name (e.g., 'Alpha')
VSP_CL_COL_ACTUAL = 'CL'        # Actual CL column name (e.g., 'CL')
VSP_CD_COL_ACTUAL = 'CDtot'     # Actual CD column name (e.g., 'CDtot', 'CD0', 'CD')

# For VLM .polar file
VLM_AOA_COL_ACTUAL = 'AoA'    # Actual AoA column name (e.g., 'Alpha')
VLM_CL_COL_ACTUAL = 'CL'        # Actual CL column name (e.g., 'CL')
VLM_CD_COL_ACTUAL = 'CDtot'        # Actual CD column name (e.g., 'CDtot', 'CD0', 'CD')


# --- USER: Define your Aspect Ratio Cases and File Paths Here ---
aspect_ratio_cases = [
    {
        'ar_label': 'AR_1.5', 
        'exp_file': 'AR_1.5_CL_CD.csv',
        'vsp_file': r'C:\Users\tomsp\OpenVSP2\OpenVSP-3.43.0-win64\validation2_VSPGeom.polar',
        'vlm_file': r'C:\Users\tomsp\OpenVSP2\OpenVSP-3.43.0-win64\validation2_DegenGeom.polar'
    },
    # Add more cases as needed
]

def read_experimental_data_for_cl_cd_plot(filepath,
                                          aoa_col_actual, cl_col_actual, cd_col_actual,
                                          aoa_col_idx, cl_col_idx, cd_col_idx):
    """
    Reads experimental AoA, CL, and CD data from a CSV file.
    Outputs DataFrame with standardized column names: 'AoA', 'CL', 'CD'.
    """
    if not filepath or not os.path.exists(filepath):
        print(f"Warning: Experimental CSV file not found or path not specified: '{filepath}'.")
        return None
    try:
        df = pd.read_csv(filepath, comment='#', delimiter=',')
        df.columns = [col.strip() for col in df.columns]

        required_cols_actual = [aoa_col_actual, cl_col_actual, cd_col_actual]
        standard_cols = ['AoA', 'CL', 'CD'] # Standardized output names

        if all(col in df.columns for col in required_cols_actual):
            df_selected = df[required_cols_actual].copy()
            df_selected.columns = standard_cols
        else:
            print(f"Warning: One or more headers ({required_cols_actual}) not in '{filepath}'. Headers: {df.columns.tolist()}")
            required_indices = [aoa_col_idx, cl_col_idx, cd_col_idx]
            if len(df.columns) >= max(required_indices) + 1:
                print(f"Attempting to read by column indices ({required_indices}) for '{filepath}'.")
                df_selected = pd.read_csv(filepath, comment='#', delimiter=',', header=None, skiprows=1, usecols=required_indices, names=standard_cols)
            else:
                print(f"Error: File '{filepath}' does not have enough columns for index-based reading of AoA, CL, and CD.")
                return None
        
        for col in standard_cols:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        df_selected.dropna(subset=standard_cols, inplace=True)

        if df_selected.empty:
            print(f"Warning: No valid numeric data found in '{filepath}' after attempting to read columns for AoA, CL, CD.")
            return None
        
        print(f"Successfully read Experimental data (AoA, CL, CD) from '{filepath}'.")
        return df_selected.sort_values(by='AoA').reset_index(drop=True)

    except Exception as e:
        print(f"Error reading Experimental CSV '{filepath}': {e}")
        return None


def read_polar_data_for_cl_cd_plot(filepath, aoa_col_actual, cl_col_actual, cd_col_actual, source_name="Polar"):
    """
    Reads a .polar file (VSPAERO, VLM), expecting AoA, CL, and CD data.
    Selects specified columns and returns them as 'AoA', 'CL', 'CD'.
    """
    if not filepath or not os.path.exists(filepath):
        print(f"Warning: {source_name} polar file not found or path not specified: '{filepath}'.")
        return None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header_line_index = -1
        for i, line in enumerate(lines):
            if not line.strip().startswith('#') and line.strip():
                header_line_index = i
                break
        
        if header_line_index == -1:
            print(f"Error: Could not find header line in {source_name} polar file '{filepath}'.")
            return None

        data_io = StringIO("".join(lines[header_line_index:]))
        df_raw = pd.read_csv(data_io, delim_whitespace=True)
        df_raw.columns = [col.replace('#', '').strip() for col in df_raw.columns]

        required_cols_actual = {
            'AoA': aoa_col_actual,
            'CL': cl_col_actual,
            'CD': cd_col_actual
        }
        standard_cols_output = ['AoA', 'CL', 'CD']
        
        df_selected_data = {}
        for standard_name, actual_name in required_cols_actual.items():
            if actual_name not in df_raw.columns:
                print(f"Error: Required column '{actual_name}' (for {standard_name}) not found in {source_name} file '{filepath}'. Available: {df_raw.columns.tolist()}")
                return None
            df_selected_data[standard_name] = df_raw[actual_name]
        
        df_selected = pd.DataFrame(df_selected_data)

        for col in standard_cols_output:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        df_selected.dropna(subset=standard_cols_output, inplace=True)
        
        if df_selected.empty:
            print(f"Warning: No valid numeric data after selecting AoA, CL, CD columns from {source_name} file '{filepath}'.")
            return None

        print(f"Successfully read {source_name} data (AoA, CL, CD) from '{filepath}'.")
        return df_selected.sort_values(by='AoA').reset_index(drop=True)

    except Exception as e:
        print(f"Error reading {source_name} polar file '{filepath}': {e}")
        return None
def calculate_percentage_error(computed_values, experimental_values):
    """Calculates percentage error: (|computed - experimental| / |experimental|) * 100."""
    computed_values = np.asarray(computed_values)
    experimental_values = np.asarray(experimental_values)
    
    # Initialize error array with NaNs for cases where division by zero might occur
    # or where experimental value is zero and computed is not.
    percentage_error = np.full_like(experimental_values, np.nan, dtype=float)
    
    # Mask for non-zero experimental values
    non_zero_exp_mask = experimental_values != 0
    percentage_error[non_zero_exp_mask] = \
        (np.abs(computed_values[non_zero_exp_mask] - experimental_values[non_zero_exp_mask]) /
         np.abs(experimental_values[non_zero_exp_mask])) * 100
    
    # Mask for cases where experimental is zero AND computed is also zero (error is 0%)
    both_zero_mask = (experimental_values == 0) 
    percentage_error[both_zero_mask] = 0
    
    return percentage_error


# --- Main Script ---
def main():
    if not aspect_ratio_cases:
        print("No aspect ratio cases defined. Configure 'aspect_ratio_cases' list.")
        return

    all_accuracy_summary = [] # For VSPAERO vs Exp accuracy

    for case_info in aspect_ratio_cases:
        ar_label = case_info['ar_label']
        exp_filepath = case_info.get('exp_file')
        vsp_filepath = case_info.get('vsp_file')
        vlm_filepath = case_info.get('vlm_file') # VLM is now a .polar file

        print(f"\n--- Processing Case: {ar_label} ---")

        # 1. Read Experimental Data (AoA, CL, CD)
        df_exp = None
        if exp_filepath:
            df_exp = read_experimental_data_for_cl_cd_plot(exp_filepath,
                                                           EXP_AOA_COL_ACTUAL, EXP_CL_COL_ACTUAL, EXP_CD_COL_ACTUAL,
                                                           EXP_AOA_COL_IDX, EXP_CL_COL_IDX, EXP_CD_COL_IDX)
        
        if df_exp is None or df_exp.empty:
            print(f"Experimental data for {ar_label} is missing or couldn't be loaded. Skipping this case.")
            continue

        # Initialize df_accuracy_case with experimental data
        # These are the 'ground truth' values for comparison.
        df_accuracy_case = df_exp.rename(columns={'AoA': 'AoA_Exp', 'CL': 'CL_Exp', 'CD': 'CD_Exp'})

        # --- VSPAERO Data Processing (Panel Method) ---
        df_vsp = None
        if vsp_filepath:
            df_vsp = read_polar_data_for_cl_cd_plot(vsp_filepath, 
                                                    VSP_AOA_COL_ACTUAL, VSP_CL_COL_ACTUAL, VSP_CD_COL_ACTUAL, 
                                                    source_name="VSPAERO")
        
        if df_vsp is not None and not df_vsp.empty:
            # Interpolate VSPAERO CL and CD to experimental AoA points for accuracy calculation
            cl_vsp_interp = np.interp(df_accuracy_case['AoA_Exp'], df_vsp['AoA'], df_vsp['CL'])
            cd_vsp_interp = np.interp(df_accuracy_case['AoA_Exp'], df_vsp['AoA'], df_vsp['CD'])
            
            df_accuracy_case['CL_VSPAERO_Interp'] = cl_vsp_interp
            df_accuracy_case['CD_VSPAERO_Interp'] = cd_vsp_interp

            # CL Accuracy vs Experimental
            df_accuracy_case['Abs_Error_CL_VSP'] = np.abs(df_accuracy_case['CL_VSPAERO_Interp'] - df_accuracy_case['CL_Exp'])
            df_accuracy_case['Perc_Error_CL_VSP (%)'] = np.where(
                df_accuracy_case['CL_Exp'] == 0,
                np.where(df_accuracy_case['Abs_Error_CL_VSP'] == 0, 0, np.nan),
                (df_accuracy_case['Abs_Error_CL_VSP'] / np.abs(df_accuracy_case['CL_Exp'])) * 100
            ).round(2)

            # CD Accuracy vs Experimental
            df_accuracy_case['Abs_Error_CD_VSP'] = np.abs(df_accuracy_case['CD_VSPAERO_Interp'] - df_accuracy_case['CD_Exp'])
            df_accuracy_case['Perc_Error_CD_VSP (%)'] = np.where(
                df_accuracy_case['CD_Exp'] == 0,
                np.where(df_accuracy_case['Abs_Error_CD_VSP'] == 0, 0, np.nan),
                (df_accuracy_case['Abs_Error_CD_VSP'] / np.abs(df_accuracy_case['CD_Exp'])) * 100
            ).round(2)
        else:
            print(f"VSPAERO data not loaded or empty for {ar_label}. VSPAERO accuracy columns will be NaN.")
            for col in ['CL_VSPAERO_Interp', 'CD_VSPAERO_Interp', 
                        'Abs_Error_CL_VSP', 'Perc_Error_CL_VSP (%)',
                        'Abs_Error_CD_VSP', 'Perc_Error_CD_VSP (%)']:
                df_accuracy_case[col] = np.nan
        
        df_accuracy_case['AR_Label'] = ar_label
        all_accuracy_summary.append(df_accuracy_case)

        print(f"\nAccuracy Comparison (VSPAERO vs Experimental) for {ar_label}:")
        cols_to_print = ['AoA_Exp', 'CL_Exp', 'CD_Exp', 
                         'CL_VSPAERO_Interp', 'Abs_Error_CL_VSP', 'Perc_Error_CL_VSP (%)',
                         'CD_VSPAERO_Interp', 'Abs_Error_CD_VSP', 'Perc_Error_CD_VSP (%)']
        cols_to_print = [col for col in cols_to_print if col in df_accuracy_case.columns and df_accuracy_case[col].notna().any()]
        if len(cols_to_print) > 3: # Only print if there's more than just Exp data
             print(df_accuracy_case[cols_to_print])
        else:
            print("No VSPAERO data available for accuracy comparison.")


        # --- VLM Data Processing (for plotting only, read from .polar) ---
        df_vlm = None
        if vlm_filepath:
            df_vlm = read_polar_data_for_cl_cd_plot(vlm_filepath, 
                                                    VLM_AOA_COL_ACTUAL, VLM_CL_COL_ACTUAL, VLM_CD_COL_ACTUAL, 
                                                    source_name="VLM")
        if df_vlm is None or df_vlm.empty:
            print(f"VLM data not loaded or empty for {ar_label}. VLM will not be plotted.")
        

        # --- Plotting CL vs CD ---
        plt.figure(figsize=(10, 8))
        
        # Plot Experimental CL vs CD
        plt.plot(df_exp['CD'], df_exp['CL'], marker='o', linestyle='--', label=f'Experimental ({ar_label})', color='black', zorder=3)

        # Plot VSPAERO CL vs CD (using its own CD and CL values)
        if df_vsp is not None and not df_vsp.empty:
            plt.plot(df_vsp['CD'], df_vsp['CL'], marker='^', linestyle='-', label=f'Panel ({ar_label})', color='blue', zorder=2)
        
        # Plot VLM CL vs CD (using its own CD and CL values)
        if df_vlm is not None and not df_vlm.empty:
            plt.plot(df_vlm['CD'], df_vlm['CL'], marker='s', linestyle='-.', label=f'VLM ({ar_label})', color='green', zorder=1)
        
        plt.xlabel('Drag Coefficient (CD)')
        plt.ylabel('Lift Coefficient (CL)')
        plt.title(f'CL vs. CD Comparison for {ar_label}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig(f'plot_CL_vs_CD_{ar_label}.png')
        plt.show()

    if all_accuracy_summary:
        df_all_summary_combined = pd.concat(all_accuracy_summary).reset_index(drop=True)
        print("\n--- Combined Accuracy Summary (VSPAERO vs Experimental) Across All Aspect Ratios ---")
        print(df_all_summary_combined.to_string()) # Use to_string for better console display of wide tables
        
        if 'Perc_Error_CL_VSP (%)' in df_all_summary_combined.columns:
            avg_error_cl_vsp = df_all_summary_combined.groupby('AR_Label')['Perc_Error_CL_VSP (%)'].mean().round(2)
            print("\nAverage Percentage Error (CL_VSP) per Aspect Ratio:")
            print(avg_error_cl_vsp)
        if 'Perc_Error_CD_VSP (%)' in df_all_summary_combined.columns:
            avg_error_cd_vsp = df_all_summary_combined.groupby('AR_Label')['Perc_Error_CD_VSP (%)'].mean().round(2)
            print("\nAverage Percentage Error (CD_VSP) per Aspect Ratio:")
            print(avg_error_cd_vsp)
        # df_all_summary_combined.to_csv('combined_cl_cd_accuracy_summary.csv', index=False)
        print(f"\n--- Processing Error Plot for Case: {ar_label} ---")

        df_exp = None
        if exp_filepath:
            df_exp = read_experimental_data_for_cl_cd_plot(exp_filepath,
                                                     EXP_AOA_COL_ACTUAL, EXP_CL_COL_ACTUAL, EXP_CD_COL_ACTUAL,
                                                     EXP_AOA_COL_IDX, EXP_CL_COL_IDX, EXP_CD_COL_IDX)
        if df_exp is None or df_exp.empty:
            print(f"Experimental data for {ar_label} is essential for error plotting and is missing. Skipping this case.")

        # Prepare a DataFrame for errors, using experimental AoA as index
        aoa_exp_points = df_exp['AoA'].values
        error_data = pd.DataFrame({'AoA': aoa_exp_points})

        # --- VSPAERO (Panel) Error Calculation ---
        df_vsp = None
        if vsp_filepath:
            df_vsp = read_polar_data_for_cl_cd_plot(vsp_filepath, VSP_AOA_COL_ACTUAL, VSP_CL_COL_ACTUAL, VSP_CD_COL_ACTUAL, "VSPAERO")
        
        if df_vsp is not None and not df_vsp.empty:
            cl_vsp_interp = np.interp(aoa_exp_points, df_vsp['AoA'], df_vsp['CL'])
            cd_vsp_interp = np.interp(aoa_exp_points, df_vsp['AoA'], df_vsp['CD'])
            error_data['CL_Error_VSP (%)'] = calculate_percentage_error(cl_vsp_interp, df_exp['CL'])
            error_data['CD_Error_VSP (%)'] = calculate_percentage_error(cd_vsp_interp, df_exp['CD'])
        else:
            print(f"VSPAERO data not loaded for {ar_label}. VSP errors will be NaN.")
            error_data['CL_Error_VSP (%)'] = np.nan
            error_data['CD_Error_VSP (%)'] = np.nan

        # --- VLM Error Calculation ---
        df_vlm = None
        if vlm_filepath:
            df_vlm = read_polar_data_for_cl_cd_plot(vlm_filepath, VLM_AOA_COL_ACTUAL, VLM_CL_COL_ACTUAL, VLM_CD_COL_ACTUAL, "VLM")

        if df_vlm is not None and not df_vlm.empty:
            cl_vlm_interp = np.interp(aoa_exp_points, df_vlm['AoA'], df_vlm['CL'])
            cd_vlm_interp = np.interp(aoa_exp_points, df_vlm['AoA'], df_vlm['CD'])
            error_data['CL_Error_VLM (%)'] = calculate_percentage_error(cl_vlm_interp, df_exp['CL'])
            error_data['CD_Error_VLM (%)'] = calculate_percentage_error(cd_vlm_interp, df_exp['CD'])
        else:
            print(f"VLM data not loaded for {ar_label}. VLM errors will be NaN.")
            error_data['CL_Error_VLM (%)'] = np.nan
            error_data['CD_Error_VLM (%)'] = np.nan
            
        print(f"\nCalculated Errors for {ar_label}:")
        print(error_data.to_string())

        # --- Plotting Errors ---
        plt.figure(figsize=(12, 8))
        plot_has_data = False

        if 'CL_Error_VSP (%)' in error_data and error_data['CL_Error_VSP (%)'].notna().any():
            plt.plot(error_data['AoA'], error_data['CL_Error_VSP (%)'], linestyle='-', label='Panel $C_L$ Error (%)', color='blue')
            plot_has_data = True
        if 'CD_Error_VSP (%)' in error_data and error_data['CD_Error_VSP (%)'].notna().any():
            plt.plot(error_data['AoA'], error_data['CD_Error_VSP (%)'],  linestyle='-', label='Panel $C_D$ Error (%)', color='red')
            plot_has_data = True
        if 'CL_Error_VLM (%)' in error_data and error_data['CL_Error_VLM (%)'].notna().any():
            plt.plot(error_data['AoA'], error_data['CL_Error_VLM (%)'], linestyle='--', label='VLM $C_L$ Error (%)', color='blue')
            plot_has_data = True
        if 'CD_Error_VLM (%)' in error_data and error_data['CD_Error_VLM (%)'].notna().any():
            plt.plot(error_data['AoA'], error_data['CD_Error_VLM (%)'], linestyle='--', label='VLM $C_D$ Error (%)', color='red')
            plot_has_data = True
        
        if plot_has_data:
            plt.xlabel('$\\alpha$ ($\\degree$)')
            plt.ylabel('Percentage Error (%)')
            plt.title(f'$C_L$ and $C_D$ Percentage Error vs. AoA for $AR$ 1.5\n(Compared to Experimental Data)')
            plt.legend()
            plt.grid(True)
            plt.axhline(0, color='black', linewidth=0.5, linestyle=':') # Zero error line
            plt.tight_layout()
            # plt.savefig(f'error_plot_{ar_label}.png')
            plt.show()
        else:
            print(f"No error data available to plot for {ar_label}.")
            plt.close() # Close empty figure

if __name__ == '__main__':
    main()