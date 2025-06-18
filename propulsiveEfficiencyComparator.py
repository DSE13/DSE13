import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from scipy.optimize import curve_fit
import warnings

plt.style.use("paper.mplstyle")

pt = 1./72.27

jour_sizes = {"PRD": {"onecol": 483.69687*pt, "halfcol": 241.84843*pt, "quartercol": 120.92421*pt},
             }

my_width = jour_sizes["PRD"]["halfcol"]
golden = (1 + 5 ** 0.5) / 2

# Constants
MPH_TO_MPS = 0.44704
TARGET_THRUST_N = 0.25 # As per original script
CRUISE_SPEED_MPS = 20.0    # As per your latest file context
CRUISE_SPEED_MPH = CRUISE_SPEED_MPS / MPH_TO_MPS
INCH_TO_METER = 0.0254

def parse_prop_data_file(filepath):
    """
    Parses a propeller data file and extracts data for each RPM.

    Args:
        filepath (str): The path to the propeller data file.

    Returns:
        dict: A dictionary where keys are RPMs (int) and values are
              pandas DataFrames containing the performance data for that RPM.
    """
    all_rpm_data = {}
    current_rpm = None
    data_lines = []
    header = None
    
    # Define a more robust column name mapping if needed, or use fixed indices
    # For now, we'll try to auto-detect and clean up
    column_names_map = {
        "V(mph)": "V_mph", "J(Adv_Ratio)": "J", "Pe": "Pe", "Ct": "Ct", "Cp": "Cp",
        "PWR(Hp)": "PWR_Hp", "Torque(In-Lbf)": "Torque_InLbf", "Thrust(Lbf)": "Thrust_Lbf",
        "PWR(W)": "PWR_W", "Torque(N-m)": "Torque_Nm", "Thrust(N)": "Thrust_N",
        "THR/PWR(g/W)": "THR_PWR_gW", "Mach": "Mach", "Reyn": "Reyn", "FOM": "FOM"
    }
    # Alternative based on observed headers (more flexible)
    # V J Pe Ct Cp PWR Torque Thrust PWR Torque Thrust THR/PWR Mach Reyn FOM
    # (mph) (Adv_Ratio) - - - (Hp) (In-Lbf) (Lbf) (W) (N-m) (N) (g/W) - - -
    
    # Tentative column names, will be refined by header
    expected_cols_clean = [
        "V_mph", "J", "Pe", "Ct", "Cp", 
        "PWR_Hp", "Torque_InLbf", "Thrust_Lbf", 
        "PWR_W", "Torque_Nm", "Thrust_N", 
        "THR_PWR_gW", "Mach_tip", "Reyn_75", "FOM"
    ]

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_data_block = False
    for line in lines:
        line = line.strip()
        if not line: # End of a data block
            if in_data_block and data_lines and header:
                try:
                    df = pd.DataFrame(data_lines, columns=header)
                    # Convert all columns to numeric, coercing errors
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(inplace=True) # Remove rows with parsing errors (like incomplete last lines)
                    all_rpm_data[current_rpm] = df
                except Exception as e:
                    print(f"Error processing data for RPM {current_rpm} in {filepath}: {e}")
                data_lines = []
                in_data_block = False
            continue

        rpm_match = re.match(r"PROP RPM =\s*(\d+)", line)
        if rpm_match:
            if in_data_block and data_lines and header: # Save previous block if any
                try:
                    df = pd.DataFrame(data_lines, columns=header)
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(inplace=True)
                    all_rpm_data[current_rpm] = df
                except Exception as e:
                    print(f"Error processing data for RPM {current_rpm} (before new RPM) in {filepath}: {e}")

            current_rpm = int(rpm_match.group(1))
            data_lines = []
            header = None
            in_data_block = False # Reset until header is found for new RPM
            continue

        # Try to identify the header line for the data table
        # This is heuristic, might need adjustment if format varies
        if current_rpm is not None and "J" in line and "Pe" in line and "Ct" in line and not in_data_block:
            # Try to split header and clean up names
            raw_header = re.split(r'\s{2,}', line) # Split on 2 or more spaces
            # A bit of manual cleaning for common patterns
            cleaned_header = []
            for h_item in raw_header:
                h_item = h_item.replace('(','').replace(')','').replace('/','_').replace('-','_').replace(' ','_')
                if h_item == "V": h_item = "V_mph" # Distinguish from V_mps
                if h_item == "J_Adv_Ratio": h_item = "J"
                if h_item == "Pe": h_item = "Pe"
                if h_item == "PWR_Hp": h_item = "PWR_Hp"
                if h_item == "Torque_In_Lbf": h_item = "Torque_InLbf"
                if h_item == "Thrust_Lbf": h_item = "Thrust_Lbf"
                if h_item == "PWR_W": h_item = "PWR_W"
                if h_item == "Torque_N_m": h_item = "Torque_Nm"
                if h_item == "Thrust_N": h_item = "Thrust_N"
                if h_item == "THR_PWR_g_W": h_item = "THR_PWR_gW"
                if h_item == "Mach": h_item = "Mach_prop_tip"
                if h_item == "Reyn": h_item = "Reyn_75pct"
                cleaned_header.append(h_item)
            
            # Ensure we have the standard number of columns we expect data for
            if len(cleaned_header) == len(expected_cols_clean):
                 header = expected_cols_clean # Use our cleaned standard names
            else: # Fallback if auto-detection is tricky, rely on fixed count
                print(f"Warning: Header mismatch for RPM {current_rpm} in {filepath}. Detected: {cleaned_header}")
                header = expected_cols_clean # Fallback to standard names
            
            in_data_block = True
            # Skip the unit line that often follows the header
            # For example: (mph) (Adv_Ratio) - - - (Hp) (In-Lbf) (Lbf) (W) (N-m) (N) (g/W) - - -
            # This is handled by the next iteration not matching data pattern if it's units
            continue 

        if in_data_block and header:
            # Data lines have fixed-width like structure or space separated
            # Try splitting by multiple spaces, robust to some inconsistencies
            values = re.split(r'\s+', line.strip())
            if len(values) == len(header): # Only process if number of values matches header
                data_lines.append(values)
            elif len(values) > 0 and all(v.replace('.', '', 1).isdigit() or v == '-' for v in values):
                 # Handle cases like the last line of PER3_4x33E at 10000 RPM
                 if len(values) < len(header):
                    print(f"Warning: Incomplete data line for RPM {current_rpm} in {filepath}: {line.strip()}. Padding with NaNs.")
                    values.extend([np.nan] * (len(header) - len(values)))
                    data_lines.append(values)
                 # else:
                 # print(f"Skipping malformed data line for RPM {current_rpm} in {filepath}: {line.strip()}")

    # Process the last block if file ends without a blank line
    if in_data_block and data_lines and header:
        try:
            df = pd.DataFrame(data_lines, columns=header)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            all_rpm_data[current_rpm] = df
        except Exception as e:
            print(f"Error processing final data block for RPM {current_rpm} in {filepath}: {e}")
            
    return all_rpm_data

def find_rpm_for_thrust(prop_data, target_thrust_n, target_speed_mph):
    """
    Finds the RPM and corresponding data that provides thrust closest
    to target_thrust_n at a speed closest to target_speed_mph.
    If no data near target_speed_mph, uses static thrust (V_mph=0).
    """
    best_rpm = None
    min_thrust_diff = float('inf')
    best_df_for_rpm = None

    if not prop_data:
        raise ValueError("Prop data is empty.")

    for rpm, df in prop_data.items():
        if 'V_mph' not in df.columns or 'Thrust_N' not in df.columns:
            print(f"RPM {rpm} DataFrame missing V_mph or Thrust_N columns. Skipping.")
            continue

        # Try to find data point near target speed
        df_at_speed = df.iloc[(df['V_mph'] - target_speed_mph).abs().argsort()[:1]]
        
        if not df_at_speed.empty:
            actual_v_mph = df_at_speed['V_mph'].iloc[0]
            thrust_at_v = df_at_speed['Thrust_N'].iloc[0]
            
            # Check if the found speed is reasonably close to target
            if abs(actual_v_mph - target_speed_mph) < 5: # Allow 5 MPH tolerance
                current_thrust_diff = abs(thrust_at_v - target_thrust_n)
                if current_thrust_diff < min_thrust_diff:
                    min_thrust_diff = current_thrust_diff
                    best_rpm = rpm
                    best_df_for_rpm = df
                    print(f"RPM {rpm}: Thrust {thrust_at_v:.2f}N @ {actual_v_mph:.2f} MPH (Target ~{target_speed_mph:.2f} MPH). Diff: {current_thrust_diff:.3f}")
                continue # Prioritize data at cruise speed

        # # Fallback to static thrust if no good match at cruise speed or if it's better
        # static_data = df[df['V_mph'] == 0]
        # if not static_data.empty:
        #     static_thrust = static_data['Thrust_N'].iloc[0]
        #     current_thrust_diff = abs(static_thrust - target_thrust_n)
        #      # If this static thrust is better than any found at cruise, or if none at cruise found yet
        #     if best_rpm is None or current_thrust_diff < min_thrust_diff * 0.9 : # Prefer cruise match unless static is much better
        #         min_thrust_diff = current_thrust_diff
        #         best_rpm = rpm
        #         best_df_for_rpm = df
        #         print(f"RPM {rpm}: Static Thrust {static_thrust:.2f}N. Diff: {current_thrust_diff:.3f} (Used as fallback or better static match)")


    if best_rpm is None:
        # If still no RPM found, pick the one with static thrust closest to target
        print("Warning: Could not find ideal RPM at cruise speed. Falling back to overall best static thrust match.")
        for rpm, df in prop_data.items():
            static_data = df[df['V_mph'] == 0]
            if not static_data.empty:
                static_thrust = static_data['Thrust_N'].iloc[0]
                current_thrust_diff = abs(static_thrust - target_thrust_n)
                if current_thrust_diff < min_thrust_diff:
                    min_thrust_diff = current_thrust_diff
                    best_rpm = rpm
                    best_df_for_rpm = df
        if best_rpm is None:
             raise ValueError("Could not find any RPM with valid static thrust data.")


    print(f"Selected RPM: {best_rpm} with thrust diff {min_thrust_diff:.3f} N from target {target_thrust_n} N.")
    return best_rpm, best_df_for_rpm

def quadratic_func(x, a, b, c):
    """Quadratic function for regression."""
    return a * x**2 + b * x + c

def find_rpm_for_thrust_interpolated(prop_data, target_thrust_n, target_speed_mph, prop_name_short):
    """
    Finds an optimal RPM using regression to achieve target_thrust_n at target_speed_mph.
    """
    rpms_for_interp = []
    thrusts_at_cruise_for_interp = []

    print(f"\nInterpolating thrust at {target_speed_mph:.2f} MPH for {prop_name_short} to find optimal RPM:")
    for rpm_val, df_rpm_data in sorted(prop_data.items()):
        if 'V_mph' in df_rpm_data.columns and 'Thrust_N' in df_rpm_data.columns and not df_rpm_data.empty:
            df_sorted_by_v = df_rpm_data.sort_values(by='V_mph')
            v_mph_col = df_sorted_by_v['V_mph'].values
            thrust_n_col = df_sorted_by_v['Thrust_N'].values

            if v_mph_col.min() <= target_speed_mph <= v_mph_col.max():
                thrust_at_target_speed = np.interp(target_speed_mph, v_mph_col, thrust_n_col)
                rpms_for_interp.append(rpm_val)
                thrusts_at_cruise_for_interp.append(thrust_at_target_speed)
                # print(f"  RPM {rpm_val}: Thrust @ {target_speed_mph:.2f} MPH = {thrust_at_target_speed:.3f}N")
            # else:
                # print(f"  RPM {rpm_val}: Target speed {target_speed_mph:.2f} MPH is outside data range [{v_mph_col.min():.2f}-{v_mph_col.max():.2f}] MPH. Skipping.")
        # else:
            # print(f"  RPM {rpm_val}: Missing V_mph or Thrust_N data. Skipping.")
    
    if not rpms_for_interp: # If all RPMs were skipped
        print(f"Warning for {prop_name_short}: No RPMs had data covering {target_speed_mph:.2f} MPH. Falling back to discrete RPM selection.")
        best_rpm_discrete, _ = find_rpm_for_thrust(prop_data, target_thrust_n, target_speed_mph)
        return best_rpm_discrete if best_rpm_discrete is not None else np.nan


    if len(rpms_for_interp) < 3:
        print(f"Warning for {prop_name_short}: Less than 3 data points ({len(rpms_for_interp)}) for RPM vs Thrust regression. Using point closest to target thrust.")
        # Fallback: find RPM from the (RPM, thrust_at_cruise) list whose thrust is closest to target_thrust_n
        if not thrusts_at_cruise_for_interp: # Should be caught by the above check len(rpms_for_interp)
             best_rpm_discrete, _ = find_rpm_for_thrust(prop_data, target_thrust_n, target_speed_mph)
             return best_rpm_discrete if best_rpm_discrete is not None else np.nan

        closest_idx = np.argmin(np.abs(np.array(thrusts_at_cruise_for_interp) - target_thrust_n))
        rpm_fallback = rpms_for_interp[closest_idx]
        print(f"  Fallback: RPM {rpm_fallback} gives thrust {thrusts_at_cruise_for_interp[closest_idx]:.3f}N at cruise, closest to {target_thrust_n:.2f}N.")
        return rpm_fallback

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            params, _ = curve_fit(quadratic_func, np.array(rpms_for_interp), np.array(thrusts_at_cruise_for_interp), maxfev=5000)
        a, b, c_reg = params
        print(f"Regression for {prop_name_short} (Thrust = a*RPM^2 + b*RPM + c): a={a:.2e}, b={b:.2e}, c={c_reg:.2e}")
    except RuntimeError as e:
        print(f"Warning for {prop_name_short}: Regression failed ({e}). Falling back to discrete RPM selection.")
        best_rpm_discrete, _ = find_rpm_for_thrust(prop_data, target_thrust_n, target_speed_mph)
        return best_rpm_discrete if best_rpm_discrete is not None else np.nan

    c_prime = c_reg - target_thrust_n
    
    if abs(a) < 1e-12: # Effectively linear
        if abs(b) < 1e-12: # Effectively constant
            print(f"Warning for {prop_name_short}: Regression resulted in constant thrust (~{c_reg:.3f}N). Cannot robustly solve for target RPM. Falling back.")
            closest_idx = np.argmin(np.abs(np.array(thrusts_at_cruise_for_interp) - target_thrust_n))
            return rpms_for_interp[closest_idx]
        rpm_solution = -c_prime / b
        solutions = [rpm_solution]
        print(f"Linear regression solution for {prop_name_short}: RPM = {rpm_solution:.0f}")
    else: # Quadratic
        discriminant = b**2 - 4*a*c_prime
        if discriminant < 0:
            # Vertex of parabola: -b / (2a). Thrust at vertex: quadratic_func(-b/(2a), a,b,c_reg)
            rpm_at_vertex = -b / (2 * a)
            thrust_at_vertex = quadratic_func(rpm_at_vertex, a, b, c_reg)
            print(f"Warning for {prop_name_short}: No real RPM solution from quadratic regression (discriminant {discriminant:.2e} < 0). Target thrust {target_thrust_n:.2f}N may be unachievable.")
            print(f"  Thrust curve vertex at RPM {rpm_at_vertex:.0f} gives {thrust_at_vertex:.3f}N. Selecting this RPM as best effort if within range.")
            solutions = [rpm_at_vertex] # Use the vertex RPM as the "best effort"
        else:
            sqrt_discriminant = np.sqrt(discriminant)
            rpm1 = (-b + sqrt_discriminant) / (2*a)
            rpm2 = (-b - sqrt_discriminant) / (2*a)
            solutions = [rpm1, rpm2]
            print(f"Quadratic solutions for {prop_name_short} RPM: {rpm1:.0f}, {rpm2:.0f}")

    valid_solutions = []
    min_rpm_data, max_rpm_data = min(rpms_for_interp), max(rpms_for_interp)
    # Define a reasonable RPM range, can be wider than data but not excessively so
    reasonable_min_rpm = min_rpm_data * 0.8 
    reasonable_max_rpm = max_rpm_data * 1.2

    for r_sol in solutions:
        if reasonable_min_rpm <= r_sol <= reasonable_max_rpm and r_sol > 0 :
             # Check if this solution makes sense (e.g. thrust is not wildly off due to bad part of curve)
            thrust_check = quadratic_func(r_sol, a,b,c_reg)
            if abs(thrust_check - target_thrust_n) < abs(target_thrust_n * 0.5): # Allow 50% thrust diff
                valid_solutions.append(r_sol)
            else:
                print(f"  RPM solution {r_sol:.0f} discarded (thrust {thrust_check:.2f}N far from target {target_thrust_n:.2f}N).")


    if not valid_solutions:
        print(f"Warning for {prop_name_short}: No valid/reasonable RPM solution from regression. Falling back to RPM with closest thrust from interpolated points.")
        closest_idx = np.argmin(np.abs(np.array(thrusts_at_cruise_for_interp) - target_thrust_n))
        return rpms_for_interp[closest_idx]
    elif len(valid_solutions) == 1:
        interpolated_rpm_optimal = valid_solutions[0]
    else: # Multiple valid solutions, pick one closest to the mean of the RPMs used for fitting
        mean_rpm_data = np.mean(rpms_for_interp)
        interpolated_rpm_optimal = min(valid_solutions, key=lambda r: abs(r - mean_rpm_data))
        print(f"  Multiple valid solutions: {valid_solutions}. Chose {interpolated_rpm_optimal:.0f} (closest to mean RPM {mean_rpm_data:.0f}).")

    print(f"Selected interpolated RPM for {prop_name_short}: {interpolated_rpm_optimal:.0f} RPM for target thrust {target_thrust_n:.2f}N at {target_speed_mph:.2f} MPH.")
    return interpolated_rpm_optimal


def get_interpolated_pe_j_curve(prop_data, target_rpm, prop_name_short):
    """
    Generates a Pe vs J curve for a specific target_rpm by interpolating
    between bracketing discrete RPM data.
    """
    print(f"Generating Pe vs J curve for {prop_name_short} at interpolated RPM {target_rpm:.0f}...")
    sorted_rpms = sorted(prop_data.keys())

    if not sorted_rpms:
        print(f"Error for {prop_name_short}: No RPM data for Pe-J interpolation.")
        return pd.DataFrame(columns=['J', 'Pe'])

    rpm_low, rpm_high = None, None
    if target_rpm <= sorted_rpms[0]:
        rpm_low = rpm_high = sorted_rpms[0]
        print(f"  Target RPM {target_rpm:.0f} is at/below lowest data RPM ({rpm_low}). Using data for {rpm_low} RPM.")
    elif target_rpm >= sorted_rpms[-1]:
        rpm_low = rpm_high = sorted_rpms[-1]
        print(f"  Target RPM {target_rpm:.0f} is at/above highest data RPM ({rpm_low}). Using data for {rpm_low} RPM.")
    else:
        for i in range(len(sorted_rpms) - 1):
            if sorted_rpms[i] <= target_rpm <= sorted_rpms[i+1]:
                rpm_low, rpm_high = sorted_rpms[i], sorted_rpms[i+1]
                break
    
    if rpm_low is None: # Should be caught
        print(f"Error for {prop_name_short}: Could not find bracketing RPMs. Using closest discrete RPM.")
        closest_rpm = min(sorted_rpms, key=lambda r: abs(r - target_rpm))
        return prop_data[closest_rpm][['J', 'Pe']].copy().sort_values(by='J').reset_index(drop=True)

    df_low = prop_data[rpm_low].sort_values(by='J').reset_index(drop=True)
    df_high = prop_data[rpm_high].sort_values(by='J').reset_index(drop=True)

    if rpm_low == rpm_high: # Exact match or extrapolation boundary case
        return df_low[['J', 'Pe']].copy()

    # Define a common, dense J range based on the overlap of the bracketing RPMs
    j_min_overlap = max(df_low['J'].min(), df_high['J'].min())
    j_max_overlap = min(df_low['J'].max(), df_high['J'].max())

    if j_min_overlap >= j_max_overlap:
        print(f"Warning for {prop_name_short}: No overlapping J range for RPMs {rpm_low} and {rpm_high} to interpolate Pe-J curve. Using data from RPM closest to target.")
        closest_rpm_to_target = rpm_low if abs(target_rpm - rpm_low) < abs(target_rpm - rpm_high) else rpm_high
        return prop_data[closest_rpm_to_target][['J', 'Pe']].copy().sort_values(by='J').reset_index(drop=True)
    
    j_common = np.linspace(j_min_overlap, j_max_overlap, 150) # Denser J points for smoother curve

    pe_low_at_j = np.interp(j_common, df_low['J'].values, df_low['Pe'].values, left=np.nan, right=np.nan)
    pe_high_at_j = np.interp(j_common, df_high['J'].values, df_high['Pe'].values, left=np.nan, right=np.nan)

    weight_high = (target_rpm - rpm_low) / (rpm_high - rpm_low)
    pe_target_at_j = (1.0 - weight_high) * pe_low_at_j + weight_high * pe_high_at_j
    
    interpolated_df = pd.DataFrame({'J': j_common, 'Pe': pe_target_at_j})
    interpolated_df.dropna(inplace=True)
    
    print(f"  Generated interpolated Pe-J curve for {target_rpm:.0f} RPM with {len(interpolated_df)} points.")
    if interpolated_df.empty:
         print(f"  Warning: Resulting Pe-J curve for {target_rpm:.0f} RPM is empty. Check J ranges and data.")
    return interpolated_df.sort_values(by='J').reset_index(drop=True)

# Function to extract diameter from filename like "PER3_DxPITE.txt" or "DxPITE.txt"
# Assumes diameter is the first number in a "DxP" pattern.
def get_diameter_from_filename(filename_str):
    basename = os.path.basename(filename_str)
    # Regex to find patterns like 4x3.3, 10x4.7, 4.5x4, 41x41 etc.
    # Extracts the part before 'x' or 'X'
    match = re.search(r'(\d+(?:\.\d+)?|\d+)[xX]', basename)
    if match:
        diameter_str = match.group(1)
        # Specific handling for "41x41" pattern in filename, interpreting "41" as 4.1 inches
        if diameter_str == "41" and "41x41" in basename:
            print(f"Interpreting diameter '41' from '{basename}' as 4.1 inches due to '41x41' pattern.")
            diameter_inches = 4.1
        else:
            try:
                diameter_inches = float(diameter_str)
            except ValueError:
                print(f"Warning: Could not parse diameter '{diameter_str}' from filename {basename}. Using default 4 inches.")
                return 4.0 * INCH_TO_METER
        return diameter_inches * INCH_TO_METER
    else:
        print(f"Warning: Could not reliably determine diameter from filename {basename}. Using default 4 inches.")
        return 4.0 * INCH_TO_METER # Default to 4 inches if not found

# --- Main Script ---
file_41x41E = r'C:\Users\kilianseibl\Downloads\DSE13\PER3_41x41E.txt'

D_prop1_meters = get_diameter_from_filename(file_41x41E)
prop1_basename = os.path.basename(file_41x41E)
prop1_name_short = os.path.splitext(prop1_basename)[0].replace("PER3_", "")

# Check if files exist
if not os.path.exists(file_41x41E):
    print(f"Error: File {file_41x41E} not found.")
    exit() # Ensure script exits if file not found

print(f"\nParsing {prop1_basename} (D={D_prop1_meters/INCH_TO_METER:.1f}in)...")
data_prop1 = parse_prop_data_file(file_41x41E)

if not data_prop1:
    print(f"Error: Parsing failed for {file_41x41E}. Cannot continue.")
    exit()

# Determine optimal RPM using the new interpolative method
rpm_prop1_optimal_interp = find_rpm_for_thrust_interpolated(data_prop1, TARGET_THRUST_N, CRUISE_SPEED_MPH, prop1_name_short)

# Generate the Pe vs J curve for this interpolated RPM
df_prop1_optimal_data_interp = pd.DataFrame() # Initialize as empty
if not np.isnan(rpm_prop1_optimal_interp): # Check if a valid RPM was found
    df_prop1_optimal_data_interp = get_interpolated_pe_j_curve(data_prop1, rpm_prop1_optimal_interp, prop1_name_short)
else:
    print(f"Could not determine a valid interpolated RPM for {prop1_name_short}. Plot will not show interpolated curve.")


# --- Plotting Pe vs J for various RPMs ---
plt.figure(figsize=(my_width, my_width / golden)) # Slightly larger figure

# Define colors for original RPM curves
num_orig_rpms_prop1 = len(data_prop1)
colors_orig_prop1 = plt.cm.Greys(np.linspace(0.4, 0.8, num_orig_rpms_prop1 if num_orig_rpms_prop1 > 0 else 1))

# Plot original discrete RPM curves (dimmed)
if data_prop1:
    print(f"\nPlotting original {len(data_prop1)} discrete RPM curves for {prop1_name_short}...")
    for i, (rpm_val, df_rpm_data) in enumerate(sorted(data_prop1.items())):
        if 'J' in df_rpm_data.columns and 'Pe' in df_rpm_data.columns and not df_rpm_data.empty:
            df_sorted = df_rpm_data.sort_values(by='J').reset_index(drop=True)
            plt.plot(df_sorted['J'], df_sorted['Pe'],
                     color=colors_orig_prop1[i % len(colors_orig_prop1)], 
                     linewidth=0.7, alpha=0.5, linestyle='-', zorder=2,
                     label=f'{rpm_val} RPM (orig)' if num_orig_rpms_prop1 <= 5 else None) # Label if few lines
                     # Consider labeling only a few representative original RPMs if there are many

# Plot the new interpolated optimal curve (highlighted)
interp_line_color = 'dodgerblue' 
if not df_prop1_optimal_data_interp.empty:
    plt.plot(df_prop1_optimal_data_interp['J'], df_prop1_optimal_data_interp['Pe'],
             label=f'{prop1_name_short} @ {rpm_prop1_optimal_interp:.0f} RPM (interp.)',
             color=interp_line_color, linewidth=2.2, alpha=0.9, zorder=6)
    print(f"Plotting interpolated Pe-J curve for {rpm_prop1_optimal_interp:.0f} RPM.")
else:
    print(f"Warning: Interpolated Pe-J DataFrame for {prop1_name_short} is empty. Not plotting its curve.")
    # Optional: Fallback to plotting the discrete best if interpolation failed
    # rpm_prop1_optimal_discrete, df_prop1_optimal_data_discrete = find_rpm_for_thrust(data_prop1, TARGET_THRUST_N, CRUISE_SPEED_MPH)
    # if df_prop1_optimal_data_discrete is not None and not df_prop1_optimal_data_discrete.empty:
    #     df_sorted_discrete = df_prop1_optimal_data_discrete.sort_values(by='J').reset_index(drop=True)
    #     plt.plot(df_sorted_discrete['J'], df_sorted_discrete['Pe'],
    #              label=f'{prop1_name_short} @ {rpm_prop1_optimal_discrete} RPM (discrete fallback)',
    #              color='darkorange', linewidth=1.8, alpha=0.8, zorder=5, linestyle='--')

# Mark cruise condition on the INTERPOLATED curve
if not np.isnan(rpm_prop1_optimal_interp) and D_prop1_meters > 0 and not df_prop1_optimal_data_interp.empty:
    n_rps_optimal_prop1_interp = rpm_prop1_optimal_interp / 60.0
    if n_rps_optimal_prop1_interp > 0 : # Ensure positive RPS
        J_cruise_prop1_interp = CRUISE_SPEED_MPS / (n_rps_optimal_prop1_interp * D_prop1_meters)
        pe_cruise_prop1_interp = np.interp(J_cruise_prop1_interp,
                                           df_prop1_optimal_data_interp['J'].values,
                                           df_prop1_optimal_data_interp['Pe'].values,
                                           left=np.nan, right=np.nan)

        if not np.isnan(pe_cruise_prop1_interp):
            plt.scatter([J_cruise_prop1_interp], [pe_cruise_prop1_interp], 
                        color=interp_line_color, marker='o', s=20, zorder=10, edgecolor='black',
                        label=f'Cruise (J={J_cruise_prop1_interp:.3f}, Pe={pe_cruise_prop1_interp:.3f})')
            print(f"Marking cruise point: J={J_cruise_prop1_interp:.3f}, Pe={pe_cruise_prop1_interp:.3f} at {rpm_prop1_optimal_interp:.0f} RPM.")
        else:
            print(f"Could not interpolate Pe at calculated J_cruise={J_cruise_prop1_interp:.3f} on the interpolated Pe-J curve for {prop1_name_short}.")
            print(f"  Interpolated J range: {df_prop1_optimal_data_interp['J'].min():.3f} - {df_prop1_optimal_data_interp['J'].max():.3f}")


plt.xlabel('Advance Ratio (J)')
plt.ylabel('Propulsive Efficiency (Pe)')
plt.title(f'Thrust $\\approx$ {TARGET_THRUST_N:.1f}N, Cruise = {CRUISE_SPEED_MPS:.1f} m/s')
plt.legend(loc='best')
plt.ylim(bottom=0)

all_j_values_plot = []
if data_prop1:
    for _rpm, df in data_prop1.items(): 
        if 'J' in df.columns and not df['J'].empty: all_j_values_plot.extend(df['J'].dropna().tolist())
if not df_prop1_optimal_data_interp.empty and 'J' in df_prop1_optimal_data_interp.columns and not df_prop1_optimal_data_interp['J'].empty :
    all_j_values_plot.extend(df_prop1_optimal_data_interp['J'].dropna().tolist())

if all_j_values_plot: 
    min_j_plot = min(all_j_values_plot) if any(x < 0 for x in all_j_values_plot) else 0 
    max_j_plot = max(all_j_values_plot) if all_j_values_plot else 1.5
    plt.xlim(left=min_j_plot, right=max_j_plot * 1.05)
else:
    plt.xlim(left=0, right=1.5)

plt.tight_layout()
plt.show()