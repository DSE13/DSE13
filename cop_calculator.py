import pandas as pd
import matplotlib.pyplot as plt
import math
import os # For file path operations

def calculate_and_plot_cop_from_file(file_path, x_mref_c, delimiter=None):
    """
    Reads aerodynamic data from a VSPAERO .polar or a .csv file, calculates
    the Center of Pressure (CoP), and plots CoP vs. Angle of Attack (AoA).

    Args:
        file_path (str): The path to the data file (.polar or .csv).
        x_mref_c (float): The non-dimensional moment reference point (x_mref / c_ref).
                          Example: 0.25 for quarter-chord.
        delimiter (str, optional): The delimiter to use. If None, it will try to infer.
                                   Common values: ' ' (space for .polar), ',' (comma for .csv).
    """
    # --- 1. Parse the data from the file ---
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
            return

        file_extension = os.path.splitext(file_path)[1].lower()

        if delimiter is None:
            if file_extension == '.csv':
                delimiter_to_try = ';'
                print("Detected .csv extension, trying comma delimiter.")
            else: # For .polar or other extensions, assume space/whitespace
                delimiter_to_try = r'\s+' # Regex for one or more whitespace characters
                print(f"Detected '{file_extension}' (or no specific) extension, trying whitespace delimiter.")
        else:
            delimiter_to_try = delimiter
            print(f"Using specified delimiter: '{delimiter_to_try}'")


        # Attempt to read the file
        try:
            df = pd.read_csv(file_path, sep=delimiter_to_try, header=0, comment='#', skipinitialspace=True)
        except pd.errors.ParserError as pe:
            print(f"ParserError with delimiter '{delimiter_to_try}': {pe}")
            if file_extension == '.csv' and delimiter_to_try != ';':
                print("Trying semicolon delimiter for .csv file...")
                delimiter_to_try = ';'
                df = pd.read_csv(file_path, sep=delimiter_to_try, header=0, comment='#', skipinitialspace=True)
            elif (file_extension == '.polar' or file_extension == '.dat') and delimiter_to_try != ',':
                 print("Trying comma delimiter as fallback...")
                 delimiter_to_try = ','
                 df = pd.read_csv(file_path, sep=delimiter_to_try, header=0, comment='#', skipinitialspace=True)
            else:
                raise # Re-raise the error if fallbacks didn't work or weren't applicable

        # Clean up column names (remove leading '#', strip whitespace)
        df.columns = df.columns.str.replace(r'^#\s*', '', regex=True).str.strip()

        print(f"Data loaded successfully from: {file_path} (using delimiter '{delimiter_to_try}')")
        print("Columns found:", df.columns.tolist())
        if df.empty:
            print("The file was loaded but appears to be empty or only contained comments.")
            return

    except FileNotFoundError: # Should be caught by os.path.exists earlier
        print(f"Error: The file '{file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty or has no data after skipping comments.")
        print("Please inspect the file manually.")
        return
    except Exception as e:
        print(f"Error reading or parsing file '{file_path}': {e}")
        return

    # Ensure required columns exist
    # VSPAERO polar might use CDtot, others might use CD. Let's be flexible or ask.
    # For now, assuming CDtot is the primary drag coefficient.
    # CFz is used directly as per the original formula. If CN is directly available and preferred, adjust.
    required_cols_map = {'AoA': ['AoA', 'Alpha', 'alpha'],
                         'CL': ['CL', 'Cl'],
                         'CDtot': ['CDtot', 'CD', 'Cd'], # Check for CDtot first, then CD
                         'CMy': ['CMy', 'CM', 'Cm', 'CMy_ref'],
                         'CFz': ['CFz', 'CN', 'Cn']} # CFz is directly used in xcp_c = x_mref_c - (cmy / cfz)

    actual_col_names = {}
    missing_critical_cols = []

    for key, possible_names in required_cols_map.items():
        found = False
        for name_option in possible_names:
            if name_option in df.columns:
                actual_col_names[key] = name_option
                found = True
                break
        if not found:
            missing_critical_cols.append(key)

    if missing_critical_cols:
        print(f"Error: Missing one or more critical data columns.")
        for mc in missing_critical_cols:
             print(f"  Could not find a column for '{mc}' (tried: {', '.join(required_cols_map[mc])})")
        print(f"  Available columns in the file: {df.columns.tolist()}")
        return

    print("\nUsing columns:")
    for key, val in actual_col_names.items():
        print(f"  For {key}: Using column '{val}'")


    alphas_deg = []
    xcp_c_values = [] # To store x_cp / c_ref

    # --- 2. Calculate CoP for each data point ---
    print("\nCalculating CoP...")
    print("-------------------------------------------------------------------------------------------------")
    # Adjust print header to use actual column names for clarity if needed
    print(f"{actual_col_names.get('AoA', 'AoA'):<10} | {actual_col_names.get('CL', 'CL'):<15} | {actual_col_names.get('CDtot','CDtot'):<15} | {actual_col_names.get('CMy','CMy'):<15} | {'CN (calc)':<15} | {actual_col_names.get('CFz','CFz'):<15} | {'x_cp/c_ref':<15}")
    print("-------------------------------------------------------------------------------------------------")

    for index, row in df.iterrows():
        try:
            aoa_deg = float(row[actual_col_names['AoA']])
            cl = float(row[actual_col_names['CL']])
            # Use CDtot if found, otherwise CD
            cd_col_to_use = actual_col_names.get('CDtot', actual_col_names.get('CD'))
            cdot = float(row[cd_col_to_use]) # Using CDtot as per original
            cmy = float(row[actual_col_names['CMy']])
            cfz = float(row[actual_col_names['CFz']]) # Using CFz as per original CoP formula

            aoa_rad = math.radians(aoa_deg)

            # Calculate Normal Force Coefficient (CN) for display/interest
            # The CoP formula x_cp/c = x_mref/c - CMy / CFz uses CFz directly as the "normal" force in that context
            cn_display = cl * math.cos(aoa_rad) + cdot * math.sin(aoa_rad)

            if abs(cfz) < 1e-9:  # Avoid division by zero when using CFz
                xcp_c = float('nan')
                print(f"{aoa_deg:<10.2f} | {cl:<15.6e} | {cdot:<15.6e} | {cmy:<15.6e} | {cn_display:<15.6e} | {cfz:<15.6e} | {'Undef (CFz~0)':<15}")
            else:
                xcp_c = x_mref_c - (cmy / cfz)
                print(f"{aoa_deg:<10.2f} | {cl:<15.6e} | {cdot:<15.6e} | {cmy:<15.6e} | {cn_display:<15.6e} | {cfz:<15.6e} | {xcp_c:<15.6f}")

            alphas_deg.append(aoa_deg)
            xcp_c_values.append(xcp_c)

        except ValueError as e:
            print(f"Warning: Skipping row {index + df.index.start + 1} due to data conversion error: {e}")
            print(f"  Row data: {row.to_dict()}")
            alphas_deg.append(float(row.get(actual_col_names.get('AoA', 'AoA_fallback'), float('nan'))))
            xcp_c_values.append(float('nan'))
        except KeyError as e:
            print(f"Warning: Skipping row {index + df.index.start + 1} due to missing key: {e}")
            print(f"  Row data: {row.to_dict()}")
            alphas_deg.append(float(row.get(actual_col_names.get('AoA', 'AoA_fallback'), float('nan'))))
            xcp_c_values.append(float('nan'))
        except Exception as e:
            print(f"Warning: An unexpected error occurred processing row {index + df.index.start + 1}: {e}")
            alphas_deg.append(float(row.get(actual_col_names.get('AoA', 'AoA_fallback'), float('nan'))))
            xcp_c_values.append(float('nan'))

    print("-------------------------------------------------------------------------------------------------")

    # --- 3. Plot the results ---
    if not alphas_deg or not any(not math.isnan(x) for x in xcp_c_values):
        print("\nNo valid CoP data to plot.")
        return

    plt.figure(figsize=(10, 7))
    plt.plot(alphas_deg, xcp_c_values, linestyle='-', color='b')

    plt.xlabel(" Angle of Attack, $\\alpha$ (degrees)")
    plt.ylabel("$\\frac{x_{cp}}{c_{ref}}$")
    plot_title = (f"$V = 20$ $m/s$")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', lw=0.5)
    plt.xlim(2,25)
    plt.ylim(0, 0.25)
    # plt.ylim(-0.5, 1.5) # Example y-limits
    plt.show()

# --- Main execution part ---
if __name__ == "__main__":
    polar_file_path = input("Enter the full path to your data file (.polar or .csv): ").strip()

    while not os.path.exists(polar_file_path):
        print(f"Error: File not found at '{polar_file_path}'")
        polar_file_path = input("Please re-enter the correct file path (or Ctrl+C to exit): ").strip()
        if not polar_file_path:
            print("No file path entered. Exiting.")
            exit()

    x_mref_c_input_valid = False
    x_mref_c_value = 0.25

    while not x_mref_c_input_valid:
        try:
            x_mref_c_str = input(f"Enter the non-dimensional moment reference point (x_mref / c_ref) [default: {x_mref_c_value}]: ").strip()
            if not x_mref_c_str:
                x_mref_c_input_valid = True
            else:
                x_mref_c_value = float(x_mref_c_str)
                x_mref_c_input_valid = True
        except ValueError:
            print("Invalid input. Please enter a numeric value (e.g., 0.25).")

    print(f"\nUsing Moment Reference (x_mref/c_ref): {x_mref_c_value}")
    print(f"Processing file: {polar_file_path}\n")

    calculate_and_plot_cop_from_file(polar_file_path, x_mref_c_value)