import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy.interpolate.CubicSpline not found. Stall simulation will use linear segments instead of splines.")

EPSILON = 1e-6


def get_stall_onset_data_simplified(df_sorted_unique_aoa: pd.DataFrame, 
                                    target_aoa: float,
                                    aoa_col: str, cl_col: str, cd_col: str, cm_col: str):
    if df_sorted_unique_aoa.empty:
        return None, np.nan, np.nan, np.nan

    # Find the closest AoA in the input data
    # Ensure aoa_col values are numeric for abs() and idxmin()
    df_sorted_unique_aoa[aoa_col] = pd.to_numeric(df_sorted_unique_aoa[aoa_col], errors='coerce')
    
    # Drop rows where AoA became NaN after coercion, if any, for reliable idxmin
    df_valid_aoa = df_sorted_unique_aoa.dropna(subset=[aoa_col])
    if df_valid_aoa.empty:
        print(f"Warning: No valid numeric AoA values found in get_stall_onset_data_simplified for target {target_aoa}.")
        return None, np.nan, np.nan, np.nan
        
    idx_closest = (df_valid_aoa[aoa_col] - target_aoa).abs().idxmin()
    stall_onset_row = df_valid_aoa.loc[idx_closest].copy()
    
    # Set the AoA of the stall onset row to the target AoA
    stall_onset_row[aoa_col] = target_aoa 

    # Interpolate coefficients if there are enough points
    if len(df_valid_aoa) >= 2:
        xp = df_valid_aoa[aoa_col].values
        
        def safe_interp(fp_values_series):
            # Ensure fp_values are numeric
            fp_values_numeric = pd.to_numeric(fp_values_series, errors='coerce').values
            if np.all(np.isnan(fp_values_numeric)): 
                return np.nan 
            
            valid_indices = ~np.isnan(fp_values_numeric) & ~np.isnan(xp) # Ensure both x and y are valid
            
            # Remove duplicate xp points for interpolation
            unique_xp, unique_indices = np.unique(xp[valid_indices], return_index=True)
            valid_fp_for_unique_xp = fp_values_numeric[valid_indices][unique_indices]

            if len(unique_xp) < 1: # Needs at least one point
                 return np.nan
            if len(unique_xp) == 1: # If only one unique point, return its value (constant extrapolation)
                return valid_fp_for_unique_xp[0]
                
            return np.interp(target_aoa, unique_xp, valid_fp_for_unique_xp)

        stall_onset_row[cl_col] = safe_interp(df_valid_aoa[cl_col])
        stall_onset_row[cd_col] = safe_interp(df_valid_aoa[cd_col])
        stall_onset_row[cm_col] = safe_interp(df_valid_aoa[cm_col])
            
        # Interpolate other numeric columns
        for col in df_valid_aoa.columns:
            if pd.api.types.is_numeric_dtype(df_valid_aoa[col]) and \
               col not in [aoa_col, cl_col, cd_col, cm_col] and col in stall_onset_row:
                if col in df_valid_aoa: # Ensure column exists in the DataFrame being interpolated
                    try:
                        stall_onset_row[col] = safe_interp(df_valid_aoa[col])
                    except Exception as e: 
                         # print(f"Debug: Interpolation failed for column {col}: {e}. Keeping original value from closest point.")
                         pass # Keep original value from idx_closest if interp fails for other cols
    elif len(df_valid_aoa) == 1: 
        # If only one point, use its values (already in stall_onset_row from .loc[idx_closest])
        pass 
        
    return stall_onset_row, stall_onset_row[cl_col], stall_onset_row[cd_col], stall_onset_row[cm_col]


def get_pre_stall_slopes(df_sorted_unique_aoa: pd.DataFrame, 
                         stall_onset_row: pd.Series, 
                         aoa_stall_positive: float,
                         aoa_col: str, cl_col: str, cd_col: str, cm_col: str,
                         default_cd_slope: float):
    cl_slope_at_stall, cd_slope_at_stall, cm_slope_at_stall = 0.0, default_cd_slope, 0.0
    
    # Ensure aoa_col is numeric for comparison
    df_sorted_unique_aoa[aoa_col] = pd.to_numeric(df_sorted_unique_aoa[aoa_col], errors='coerce')
    df_pre_stall_strict = df_sorted_unique_aoa[df_sorted_unique_aoa[aoa_col] < aoa_stall_positive - EPSILON].dropna(subset=[aoa_col])


    if not df_pre_stall_strict.empty:
        p_prev = df_pre_stall_strict.iloc[-1] 
        
        # Ensure coefficients are numeric for calculation
        cl_at_stall_val = pd.to_numeric(stall_onset_row[cl_col], errors='coerce')
        cd_at_stall_val = pd.to_numeric(stall_onset_row[cd_col], errors='coerce')
        cm_at_stall_val = pd.to_numeric(stall_onset_row[cm_col], errors='coerce')
        
        p_prev_cl = pd.to_numeric(p_prev[cl_col], errors='coerce')
        p_prev_cd = pd.to_numeric(p_prev[cd_col], errors='coerce')
        p_prev_cm = pd.to_numeric(p_prev[cm_col], errors='coerce')
        p_prev_aoa = pd.to_numeric(p_prev[aoa_col], errors='coerce')


        if any(pd.isna([cl_at_stall_val, cd_at_stall_val, cm_at_stall_val, 
                        p_prev_cl, p_prev_cd, p_prev_cm, p_prev_aoa])):
            print(f"Warning: NaN values encountered when calculating pre-stall slopes for {aoa_stall_positive=}. Using default slopes.")
            # print(f"Debug NaNs: cl_stall={cl_at_stall_val}, cd_stall={cd_at_stall_val}, cm_stall={cm_at_stall_val}")
            # print(f"Debug NaNs: p_prev_cl={p_prev_cl}, p_prev_cd={p_prev_cd}, p_prev_cm={p_prev_cm}, p_prev_aoa={p_prev_aoa}")
            return cl_slope_at_stall, cd_slope_at_stall, cm_slope_at_stall

        delta_aoa = aoa_stall_positive - p_prev_aoa # p_prev[aoa_col]
        if delta_aoa > EPSILON: 
            cl_slope_at_stall = (cl_at_stall_val - p_prev_cl) / delta_aoa
            cd_slope_at_stall = (cd_at_stall_val - p_prev_cd) / delta_aoa
            cm_slope_at_stall = (cm_at_stall_val - p_prev_cm) / delta_aoa
        else:
            print(f"Warning: Delta AoA for pre-stall slope calculation is too small ({delta_aoa:.2e}) for {aoa_stall_positive=}. Using default slopes.")
    else:
        print(f"Warning: No pre-stall data points found to calculate slopes for AoA < {aoa_stall_positive:.2f}. Using default slopes.")
    return cl_slope_at_stall, cd_slope_at_stall, cm_slope_at_stall


def simulate_stall_behavior_positive_aoa(
                            df_input: pd.DataFrame, 
                            aoa_stall_abs: float = 18.0,
                            cl_drop1_value: float = 0.2,
                            cl_drop1_duration_aoa: float = 10.0,
                            cl_drop2_value: float = 0.1,
                            cl_drop2_duration_aoa: float = 15.0,
                            cd_stall_multiplier_target: float = 4.0,
                            cd_target_aoa_abs: float = 40.0,
                            cd_initial_slope_post_stall: float = 0.01,
                            cm_rise_value_abs: float = 0.2,
                            cm_rise_duration_aoa: float = 5.0,
                            cm_plateau_end_aoa_abs: float = 40.0, 
                            aoa_col: str = 'AoA',
                            cl_col: str = 'CL',
                            cd_col: str = 'CDtot',
                            cm_col: str = 'CMy') -> pd.DataFrame:
    
    df_processed = df_input.copy()
    # Ensure key columns are numeric before processing
    for col in [aoa_col, cl_col, cd_col, cm_col]:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed = df_processed.dropna(subset=[aoa_col, cl_col, cd_col, cm_col]) # Drop if essential coeffs are NaN

    if df_processed.empty:
        print(f"Warning: No valid data after coercing and dropping NaNs in simulate_stall_behavior_positive_aoa. Returning original input.")
        return df_input

    simulated_rows_list = []
    aoa_stall_positive = aoa_stall_abs

    max_aoa_cl_model = aoa_stall_positive + cl_drop1_duration_aoa + cl_drop2_duration_aoa
    max_aoa_cm_model_def = aoa_stall_positive + cm_rise_duration_aoa 
    
    max_aoa_cd_model = cd_target_aoa_abs
    max_aoa_to_simulate_up_to = max(max_aoa_cl_model, max_aoa_cm_model_def, max_aoa_cd_model, cm_plateau_end_aoa_abs)
    
    simulation_step = 0.25 
    new_aoa_simulation_points = np.array([])
    if max_aoa_to_simulate_up_to > aoa_stall_positive + simulation_step / 2: 
        _start_aoa_sim = aoa_stall_positive + simulation_step 
        _end_aoa_sim = max_aoa_to_simulate_up_to + simulation_step / 2 
        potential_points = np.arange(_start_aoa_sim, _end_aoa_sim, simulation_step)
        if len(potential_points)>0:
            new_aoa_simulation_points = potential_points[potential_points > aoa_stall_positive + EPSILON]
            if max_aoa_to_simulate_up_to > aoa_stall_positive + EPSILON and \
               (not new_aoa_simulation_points.size or np.abs(new_aoa_simulation_points[-1] - max_aoa_to_simulate_up_to) > EPSILON * 10) and \
               max_aoa_to_simulate_up_to > (new_aoa_simulation_points[-1] if new_aoa_simulation_points.size else aoa_stall_positive) :
                new_aoa_simulation_points = np.append(new_aoa_simulation_points, max_aoa_to_simulate_up_to)
            new_aoa_simulation_points = np.unique(new_aoa_simulation_points)

    df_sorted_unique_aoa = df_processed.sort_values(by=aoa_col).drop_duplicates(subset=[aoa_col])
    
    stall_onset_row_base, cl_at_stall, cd_at_stall, cm_at_stall = \
        get_stall_onset_data_simplified(df_sorted_unique_aoa, aoa_stall_positive, 
                                        aoa_col, cl_col, cd_col, cm_col)

    if stall_onset_row_base is None or pd.isna(cl_at_stall) or pd.isna(cd_at_stall) or pd.isna(cm_at_stall):
        print(f"Warning: Could not determine stall onset data at AoA {aoa_stall_positive}. Returning original data for this Beta slice.")
        # print(f"Debug stall_onset: row_base is None: {stall_onset_row_base is None}, cl_nan: {pd.isna(cl_at_stall)}, cd_nan: {pd.isna(cd_at_stall)}, cm_nan: {pd.isna(cm_at_stall)}")
        return df_input 

    cl_slope_at_stall, cd_actual_initial_slope, cm_slope_at_stall = get_pre_stall_slopes(
        df_sorted_unique_aoa, stall_onset_row_base, aoa_stall_positive,
        aoa_col, cl_col, cd_col, cm_col, default_cd_slope=cd_initial_slope_post_stall)
    
    cd_coeff_B_linear = cd_actual_initial_slope
    cd_coeff_A_quadratic = 0.0
    delta_aoa_for_cd_target = cd_target_aoa_abs - aoa_stall_positive
    if abs(delta_aoa_for_cd_target) > EPSILON:
        cd_at_stall_for_mult = cd_at_stall if not pd.isna(cd_at_stall) else 0.0
        target_total_cd_increase_at_target_aoa = (cd_stall_multiplier_target - 1.0) * cd_at_stall_for_mult
        cd_increase_from_linear_at_target_aoa = cd_coeff_B_linear * delta_aoa_for_cd_target
        cd_increase_needed_from_quadratic_at_target_aoa = target_total_cd_increase_at_target_aoa - cd_increase_from_linear_at_target_aoa
        if abs(delta_aoa_for_cd_target**2) > EPSILON : 
             cd_coeff_A_quadratic = cd_increase_needed_from_quadratic_at_target_aoa / (delta_aoa_for_cd_target**2)

    cl_spline, cm_spline = None, None
    cl_val_at_spline_end = cl_at_stall 
    cm_target_plateau_value = cm_at_stall + cm_rise_value_abs 
    cm_val_for_plateau = cm_target_plateau_value
    
    if HAS_SCIPY:
        cl_knots_x_list = [aoa_stall_positive]
        cl_knots_y_list = [cl_at_stall]
        _current_aoa_cl, _current_cl_val = aoa_stall_positive, cl_at_stall
        if cl_drop1_duration_aoa > EPSILON:
            _current_aoa_cl += cl_drop1_duration_aoa; _current_cl_val -= cl_drop1_value
            cl_knots_x_list.append(_current_aoa_cl); cl_knots_y_list.append(_current_cl_val)
        if cl_drop2_duration_aoa > EPSILON:
            _current_aoa_cl += cl_drop2_duration_aoa; _current_cl_val -= cl_drop2_value
            cl_knots_x_list.append(_current_aoa_cl); cl_knots_y_list.append(_current_cl_val)
        
        unique_cl_knots_x, unique_cl_knots_y = [], []
        if cl_knots_x_list:
            last_x = cl_knots_x_list[0] - (EPSILON * 10) 
            for x, y in zip(cl_knots_x_list, cl_knots_y_list):
                if pd.isna(x) or pd.isna(y): continue # Skip NaN knots
                if x > last_x + EPSILON: 
                    unique_cl_knots_x.append(x); unique_cl_knots_y.append(y); last_x = x
                elif not unique_cl_knots_x: 
                    unique_cl_knots_x.append(x); unique_cl_knots_y.append(y); last_x = x
        
        if len(unique_cl_knots_x) >= 2: 
            try:
                cl_spline = CubicSpline(unique_cl_knots_x, unique_cl_knots_y, bc_type=((1, cl_slope_at_stall), (1, 0.0)), extrapolate=False)
                cl_val_at_spline_end = unique_cl_knots_y[-1]
            except ValueError as e: print(f"Warning: CL spline error: {e}. Knots: x={unique_cl_knots_x}, y={unique_cl_knots_y}. Linear CL."); cl_spline = None
        elif len(unique_cl_knots_x) == 1 and len(cl_knots_x_list) > 1 : 
            print(f"Warning: All CL knots at same AoA {unique_cl_knots_x[0]}. Linear CL."); cl_spline = None
        else: print(f"Warning: Not enough distinct CL knots ({len(unique_cl_knots_x)}) for spline. Linear CL."); cl_spline = None

        cm_rise_phase_end_aoa = aoa_stall_positive + cm_rise_duration_aoa
        if cm_rise_duration_aoa > EPSILON and not pd.isna(cm_at_stall) and not pd.isna(cm_target_plateau_value): 
            cm_knots_x_list_rise = [aoa_stall_positive, cm_rise_phase_end_aoa]
            cm_knots_y_list_rise = [cm_at_stall, cm_target_plateau_value]
            if (cm_rise_phase_end_aoa - aoa_stall_positive) > EPSILON:
                try:
                    cm_spline = CubicSpline(cm_knots_x_list_rise, cm_knots_y_list_rise, 
                                            bc_type=((1, cm_slope_at_stall), (1, 0.0)), 
                                            extrapolate=False)
                except ValueError as e:
                    print(f"Warning: CM spline creation error (rise phase): {e}. Knots x={cm_knots_x_list_rise}, y={cm_knots_y_list_rise}. Using linear CM model.")
                    cm_spline = None 
            else: 
                print(f"CM rise duration ({cm_rise_duration_aoa:.2f} deg) too small for spline. Using step/linear CM model.")
                cm_spline = None
        else: 
            if pd.isna(cm_at_stall) or pd.isna(cm_target_plateau_value):
                 print("CM at stall or target plateau value is NaN. CM spline not created. Using linear/step.")
            elif cm_rise_duration_aoa <= EPSILON:
                print("CM rise duration is zero or negative. CM will step to plateau. Spline not used for rise.")
            cm_spline = None

    for aoa_curr in new_aoa_simulation_points:
        new_row_sr = stall_onset_row_base.copy()
        new_row_sr[aoa_col] = aoa_curr
        aoa_rel_stall = aoa_curr - aoa_stall_positive

        if cl_spline:
            if aoa_curr <= cl_spline.x[-1] + EPSILON: new_row_sr[cl_col] = cl_spline(aoa_curr)
            else: new_row_sr[cl_col] = cl_val_at_spline_end
        else: 
            _cl_val = cl_at_stall # Default to cl_at_stall if NaN or other issues
            if not pd.isna(cl_at_stall):
                if cl_drop1_duration_aoa > EPSILON and aoa_rel_stall <= cl_drop1_duration_aoa:
                    _cl_val = cl_at_stall - cl_drop1_value * (aoa_rel_stall / cl_drop1_duration_aoa)
                elif aoa_rel_stall > cl_drop1_duration_aoa : 
                    _cl_val = cl_at_stall - cl_drop1_value 
                    aoa_rel_drop1_end = aoa_rel_stall - cl_drop1_duration_aoa
                    if cl_drop2_duration_aoa > EPSILON and aoa_rel_drop1_end <= cl_drop2_duration_aoa:
                        _cl_val -= cl_drop2_value * (aoa_rel_drop1_end / cl_drop2_duration_aoa)
                    elif aoa_rel_drop1_end > cl_drop2_duration_aoa : 
                         _cl_val -= cl_drop2_value
            new_row_sr[cl_col] = _cl_val

        new_row_sr[cd_col] = (cd_at_stall if not pd.isna(cd_at_stall) else 0.0) + \
                             cd_coeff_B_linear * aoa_rel_stall + \
                             cd_coeff_A_quadratic * (aoa_rel_stall**2)

        cm_rise_phase_end_aoa_local = aoa_stall_positive + cm_rise_duration_aoa
        if cm_spline: 
            if aoa_curr <= cm_rise_phase_end_aoa_local + EPSILON: 
                new_row_sr[cm_col] = cm_spline(aoa_curr)
            else: 
                new_row_sr[cm_col] = cm_val_for_plateau
        else: 
            _cm_val = cm_val_for_plateau # Default to plateau if NaN or other issues
            if not pd.isna(cm_at_stall) and not pd.isna(cm_rise_value_abs):
                if cm_rise_duration_aoa > EPSILON and aoa_curr <= cm_rise_phase_end_aoa_local:
                    _cm_val = cm_at_stall + cm_rise_value_abs * (aoa_rel_stall / cm_rise_duration_aoa)
                # else: already set to cm_val_for_plateau if beyond or no rise
            new_row_sr[cm_col] = _cm_val
        
        simulated_rows_list.append(new_row_sr.to_dict())
    
    df_original_pre_stall = df_processed[df_processed[aoa_col] < aoa_stall_positive - EPSILON].copy()
    df_stall_point = pd.DataFrame([stall_onset_row_base.to_dict()]) 
    df_simulated_post_stall = pd.DataFrame(simulated_rows_list)

    if df_stall_point.empty and df_original_pre_stall.empty and df_simulated_post_stall.empty:
        print("Warning: All components for final DataFrame are empty in simulate_stall_behavior. Returning original input.")
        return df_input

    df_final = pd.concat([df_original_pre_stall, df_stall_point, df_simulated_post_stall], ignore_index=True)
    df_final = df_final.sort_values(by=[aoa_col]).reset_index(drop=True)
    df_final = df_final.drop_duplicates(subset=[aoa_col], keep='last') 
    
    return df_final


def mirror_aero_data_axisymmetric(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors aerodynamic data for an axisymmetric body.
    - Beta mirroring: CFy, CS, CMx, CMz flip sign. Others unchanged.
    - AoA mirroring: Standard aircraft symmetries (CL, CMy, CFz, L/D, CMx, CMz flip sign).
    """
    mirrored_rows_list = []
    
    aoa_col, beta_col = 'AoA', 'Beta'
    cl_col, cmy_col = 'CL', 'CMy'
    cfx_col, cfy_col, cfz_col = 'CFx', 'CFy', 'CFz'
    cmx_col, cmz_col = 'CMx', 'CMz'
    ld_col = 'L/D'
    cs_col = 'CS' 
    cml_col, cmm_col, cmn_col = 'CMl', 'CMm', 'CMn'
    mach_col = 'Mach' # Assuming Mach is present for uniqueness

    # Ensure all relevant columns are numeric, coercing errors to NaN
    cols_to_convert = [aoa_col, beta_col, cl_col, cmy_col, cfx_col, cfy_col, cfz_col,
                       cmx_col, cmz_col, ld_col, cs_col, cml_col, cmm_col, cmn_col,
                       'CDtot', 'CDo', 'CDi', 'E'] # Add other numeric cols
    for col in cols_to_convert:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

    original_records = df_input.to_dict('records')

    for orig_row_dict in original_records:
        beta_orig = orig_row_dict[beta_col]
        aoa_orig = orig_row_dict[aoa_col]

        # --- Scenario 1: Mirror AoA only ---
        # Generates data for (beta_orig, -aoa_orig)
        if not np.isclose(aoa_orig, 0.0):
            mod_row = orig_row_dict.copy()
            mod_row[aoa_col] = -aoa_orig
            
            # Odd functions of AoA (at beta_orig)
            if cl_col in mod_row: mod_row[cl_col] = -orig_row_dict[cl_col]
            if cmy_col in mod_row: mod_row[cmy_col] = -orig_row_dict[cmy_col]
            if cfz_col in mod_row: mod_row[cfz_col] = -orig_row_dict[cfz_col]
            if ld_col in mod_row and not pd.isna(orig_row_dict[ld_col]):
                mod_row[ld_col] = -orig_row_dict[ld_col]

            # Even functions of AoA (at beta_orig) - No change from orig_row_dict values
            # CMx, CMz, CFx, CFy, CS
            # (No explicit assignment needed as mod_row is a copy of orig_row_dict)

            # Update dependent moment coefficients based on the *potentially modified* CMx, CMy, CMz in mod_row
            if cmm_col in mod_row and cmy_col in mod_row: mod_row[cmm_col] = mod_row[cmy_col]
            if cml_col in mod_row and cmx_col in mod_row: mod_row[cml_col] = -mod_row[cmx_col]
            if cmn_col in mod_row and cmz_col in mod_row: mod_row[cmn_col] = -mod_row[cmz_col]
            
            mirrored_rows_list.append(mod_row)

        # --- Scenario 2: Mirror Beta only ---
        # Generates data for (-beta_orig, aoa_orig)
        if not np.isclose(beta_orig, 0.0):
            mod_row = orig_row_dict.copy()
            mod_row[beta_col] = -beta_orig

            # Odd functions of Beta (at aoa_orig)
            if cfy_col in mod_row: mod_row[cfy_col] = -orig_row_dict[cfy_col]
            if cs_col in mod_row and not pd.isna(orig_row_dict[cs_col]):
                mod_row[cs_col] = -orig_row_dict[cs_col]
            if cmx_col in mod_row: mod_row[cmx_col] = -orig_row_dict[cmx_col]
            if cmz_col in mod_row: mod_row[cmz_col] = -orig_row_dict[cmz_col]
            
            # Even functions of Beta (at aoa_orig) - No change from orig_row_dict values
            # CL, CMy, CFz, L/D, CFx
            
            # Update dependent moment coefficients
            if cmm_col in mod_row and cmy_col in mod_row: mod_row[cmm_col] = mod_row[cmy_col]
            if cml_col in mod_row and cmx_col in mod_row: mod_row[cml_col] = -mod_row[cmx_col]
            if cmn_col in mod_row and cmz_col in mod_row: mod_row[cmn_col] = -mod_row[cmz_col]

            mirrored_rows_list.append(mod_row)

        # --- Scenario 3: Mirror both AoA and Beta ---
        # Generates data for (-beta_orig, -aoa_orig)
        if not np.isclose(aoa_orig, 0.0) and not np.isclose(beta_orig, 0.0):
            mod_row = orig_row_dict.copy()
            mod_row[aoa_col] = -aoa_orig
            mod_row[beta_col] = -beta_orig

            # CL: Odd in AoA, Even in Beta -> Net Odd
            if cl_col in mod_row: mod_row[cl_col] = -orig_row_dict[cl_col]
            # CMy: Odd in AoA, Even in Beta -> Net Odd
            if cmy_col in mod_row: mod_row[cmy_col] = -orig_row_dict[cmy_col]
            # CFz: Odd in AoA, Even in Beta -> Net Odd
            if cfz_col in mod_row: mod_row[cfz_col] = -orig_row_dict[cfz_col]
            # L/D: Odd in AoA, Even in Beta -> Net Odd
            if ld_col in mod_row and not pd.isna(orig_row_dict[ld_col]):
                mod_row[ld_col] = -orig_row_dict[ld_col]
            
            # CFy: Even in AoA, Odd in Beta -> Net Odd
            if cfy_col in mod_row: mod_row[cfy_col] = -orig_row_dict[cfy_col]
            # CS: Even in AoA, Odd in Beta -> Net Odd
            if cs_col in mod_row and not pd.isna(orig_row_dict[cs_col]):
                mod_row[cs_col] = -orig_row_dict[cs_col] 

            # CMx: Even in AoA, Odd in Beta -> Net Odd
            if cmx_col in mod_row: mod_row[cmx_col] = -orig_row_dict[cmx_col]
            # CMz: Even in AoA, Odd in Beta -> Net Odd
            if cmz_col in mod_row: mod_row[cmz_col] = -orig_row_dict[cmz_col]
            
            # CFx: Even in AoA, Even in Beta -> Net Even (no change from original)
            if cfx_col in mod_row: mod_row[cfx_col] = orig_row_dict[cfx_col]

            # Update dependent moment coefficients
            if cmm_col in mod_row and cmy_col in mod_row: mod_row[cmm_col] = mod_row[cmy_col] 
            if cml_col in mod_row and cmx_col in mod_row: mod_row[cml_col] = -mod_row[cmx_col] 
            if cmn_col in mod_row and cmz_col in mod_row: mod_row[cmn_col] = -mod_row[cmz_col]
            
            mirrored_rows_list.append(mod_row)
            
    if not mirrored_rows_list:
        df_final = df_input.copy() # No new rows generated
    else:
        df_mirrored = pd.DataFrame(mirrored_rows_list)
        # It's crucial to ensure original df_input also has numeric types for concat
        for col in cols_to_convert: # Re-ensure numeric types on df_input before concat
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
        df_combined = pd.concat([df_input, df_mirrored], ignore_index=True)
    
        key_cols = [beta_col, aoa_col] 
        if mach_col in df_combined.columns and mach_col not in key_cols: # Add if not already primary key
             # Make Mach part of the uniqueness key if it exists and varies
            if df_combined[mach_col].nunique() > 1:
                 key_cols.append(mach_col)
        
        df_final = df_combined.drop_duplicates(subset=key_cols, keep='first')
    
    df_final = df_final.sort_values(by=key_cols).reset_index(drop=True)
    
    return df_final


if __name__ == '__main__':
    # Ensure correct path if stall.py and xzylo.csv are in different directories

    input_csv_file = r"C:\Users\tomsp\OpenVSP2\OpenVSP-3.43.0-win64\xzylo_V20.csv"
    
    output_stalled_csv_file = r"C:\Users\tomsp\OpenVSP2\OpenVSP-3.43.0-win64\xzylo_stall_V20.csv"
    output_mirrored_stalled_csv_file = r"C:\Users\tomsp\OpenVSP2\OpenVSP-3.43.0-win64\xzylo_mirror_V20.csv"
    
    STALL_AOA_POINT = 18.0 
    CL_DROP1_VAL, CL_DROP1_DUR = 0.2, 10.0 
    CL_DROP2_VAL, CL_DROP2_DUR = 0, 0
    CD_INITIAL_SLOPE, CD_STALL_MULTIPLIER, CD_TARGET_AOA = 0.015, 10.0, 40
    CM_RISE_VAL, CM_RISE_DUR, CM_PLATEAU_END_LABEL = 0.1, 8.0, 25
    
    print(f"Attempting to load CSV from: {input_csv_file}")
    if not os.path.exists(input_csv_file):
        print(f"ERROR: CSV file not found at {input_csv_file}")
        exit()

    try:
        df_input_data = pd.read_csv(input_csv_file, sep=',')
    except Exception as e:
        print(f"ERROR: Failed to read CSV file with pandas: {e}")
        exit()

    print(f"DEBUG: df_input_data.shape after load: {df_input_data.shape}")
    print(f"DEBUG: df_input_data.columns after load: {df_input_data.columns.tolist()}")
    
    # Ensure essential columns for filtering and processing exist
    essential_cols = ['Beta', 'AoA', 'CL', 'CDtot', 'CMy']
    for col in essential_cols:
        if col not in df_input_data.columns:
            print(f"ERROR: Essential column '{col}' is MISSING from the loaded DataFrame.")
            print("Please check your xzylo.csv file's header and separator (should be ';').")
            exit()
    
    # Convert all data to numeric where possible, coercing errors. This is crucial.
    # We'll list all expected numeric columns from your CSV.
    numeric_cols_expected = [
        'Beta', 'Mach', 'AoA', 'Re/1e6', 'CL', 'CDo', 'CDi', 'CDtot', 'CDt', 
        'CDtot_t', 'CS', 'L/D', 'E', 'CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz', 
        'CMl', 'CMm', 'CMn'
    ]
    for col in numeric_cols_expected:
        if col in df_input_data.columns:
            df_input_data[col] = pd.to_numeric(df_input_data[col], errors='coerce')
        # else:
            # print(f"Info: Column '{col}' not found in input, skipping numeric conversion for it.")

    if 'FOpt' in df_input_data.columns: 
        df_input_data['FOpt'] = df_input_data['FOpt'].fillna(0)
        # Attempt to convert FOpt to int, but handle potential non-numeric gracefully
        try:
            df_input_data['FOpt'] = df_input_data['FOpt'].astype(int)
        except ValueError:
            print("Warning: Could not convert 'FOpt' to int. It might contain non-integer values. Keeping as is or NaN.")
            df_input_data['FOpt'] = pd.to_numeric(df_input_data['FOpt'], errors='coerce')


    print(f"Input data shape after type coercion: {df_input_data.shape}")
    
    df_beta_0_original = df_input_data[np.isclose(df_input_data['Beta'].fillna(1e9), 0.0)].copy() # fillna for isclose
    if df_beta_0_original.empty and not df_input_data.empty:
        print("Warning: No data found for Beta=0 after filtering. Stall simulation might not run as expected.")
    print(f"Beta=0 data slice shape for stall simulation: {df_beta_0_original.shape}")

    df_beta_0_with_stall = df_beta_0_original # Initialize in case it's empty
    if not df_beta_0_original.empty:
        df_beta_0_with_stall = simulate_stall_behavior_positive_aoa(
            df_beta_0_original, aoa_stall_abs=STALL_AOA_POINT,
            cl_drop1_value=CL_DROP1_VAL, cl_drop1_duration_aoa=CL_DROP1_DUR,
            cl_drop2_value=CL_DROP2_VAL, cl_drop2_duration_aoa=CL_DROP2_DUR,
            cd_initial_slope_post_stall=CD_INITIAL_SLOPE, cd_stall_multiplier_target=CD_STALL_MULTIPLIER, cd_target_aoa_abs=CD_TARGET_AOA,
            cm_rise_value_abs=CM_RISE_VAL, cm_rise_duration_aoa=CM_RISE_DUR, cm_plateau_end_aoa_abs=CM_PLATEAU_END_LABEL,
            aoa_col='AoA', cl_col='CL', cd_col='CDtot', cm_col='CMy')
    
    print(f"Beta=0 data with simulated positive stall shape: {df_beta_0_with_stall.shape}")
    
    df_other_betas_original = df_input_data[~np.isclose(df_input_data['Beta'].fillna(1e9), 0.0)].copy()
    
    df_combined_stall_effect = pd.concat([df_beta_0_with_stall, df_other_betas_original], ignore_index=True)
    df_combined_stall_effect = df_combined_stall_effect.sort_values(by=['Beta', 'AoA', 'Mach']).reset_index(drop=True)
    # Use 'first' to prioritize stall-simulated data if any overlap from original input for Beta=0
    df_combined_stall_effect = df_combined_stall_effect.drop_duplicates(subset=['Beta', 'AoA', 'Mach'], keep='first') 

    print(f"Combined data (Beta=0 stalled, others original) shape: {df_combined_stall_effect.shape}")
    df_combined_stall_effect.to_csv(output_stalled_csv_file, sep=';', index=False, float_format='%.9g')
    print(f"Combined stall data (Beta=0 positive AoA stall) saved to '{output_stalled_csv_file}'")

    df_fully_mirrored_and_stalled = mirror_aero_data_axisymmetric(df_combined_stall_effect)
    print(f"Data with axisymmetric mirrored AoA/Beta and stall effects shape: {df_fully_mirrored_and_stalled.shape}")
    df_fully_mirrored_and_stalled.to_csv(output_mirrored_stalled_csv_file, sep=';', index=False, float_format='%.9g')
    print(f"Axisymmetric Mirrored and stalled data saved to '{output_mirrored_stalled_csv_file}'")

    coeffs_to_plot = {'CL': 'CL', 'CDtot': 'CDtot', 'CMy': 'CMy', 'CFx': 'CFx', 'CFy': 'CFy', 'CFz': 'CFz',
                      'CMx': 'CMx', 'CMz': 'CMz'}

    # Plotting for Beta = 0 slice
    df_plot_beta0_mirrored = df_fully_mirrored_and_stalled[np.isclose(df_fully_mirrored_and_stalled['Beta'].fillna(1e9), 0.0)].sort_values(by='AoA')
    # For comparison, use the original Beta=0 slice before any processing
    df_input_beta0_orig_plot = df_input_data[
        (np.isclose(df_input_data['Beta'].fillna(1e9), 0.0)) #& (df_input_data['AoA'] >= -EPSILON) # Original might have negative AoA, we'll filter later
    ].sort_values(by='AoA')


    # --- NEW: Combined CL and CDtot plot for Beta=0, AoA 0-25 deg ---
    cl_col_name = 'CL'
    cd_col_name = 'CDtot'

    if cl_col_name in df_plot_beta0_mirrored.columns and cd_col_name in df_plot_beta0_mirrored.columns:
        # Filter data for AoA range 0 to 25
        df_plot_beta0_clcd_proc_filtered = df_plot_beta0_mirrored[
            (df_plot_beta0_mirrored['AoA'] >= -EPSILON) & (df_plot_beta0_mirrored['AoA'] <= 25 + EPSILON)
        ].copy()

        df_plot_beta0_clcd_orig_filtered = df_input_beta0_orig_plot[
            (df_input_beta0_orig_plot['AoA'] >= -EPSILON) & (df_input_beta0_orig_plot['AoA'] <= 25 + EPSILON)
        ].copy()

        if not df_plot_beta0_clcd_proc_filtered.empty or not df_plot_beta0_clcd_orig_filtered.empty:
            fig, ax1 = plt.subplots(figsize=(16, 9))

            # Plot CL (Left Y-axis)
            color_cl_proc = 'red'
            color_cl_orig = 'blue'
            ax1.set_xlabel('AoA (degrees)', fontsize=12)
            ax1.set_ylabel(cl_col_name, color=color_cl_proc, fontsize=12)

            if cl_col_name in df_plot_beta0_clcd_proc_filtered.columns:
                ax1.plot(df_plot_beta0_clcd_proc_filtered['AoA'], df_plot_beta0_clcd_proc_filtered[cl_col_name],
                         linestyle='-', marker='none', markersize=4, color=color_cl_proc,
                         label=f'{cl_col_name} with Stall', alpha=0.7)
            if cl_col_name in df_plot_beta0_clcd_orig_filtered.columns:
                ax1.plot(df_plot_beta0_clcd_orig_filtered['AoA'], df_plot_beta0_clcd_orig_filtered[cl_col_name],
                         linestyle=':', marker='o', markersize=6, color=color_cl_orig,
                         label=f'{cl_col_name}', alpha=0.6)
            ax1.tick_params(axis='y', labelcolor=color_cl_proc)

            # Create a second Y-axis for CDtot
            ax2 = ax1.twinx()
            color_cd_proc = 'green'
            color_cd_orig = 'purple'
            ax2.set_ylabel(cd_col_name, color=color_cd_proc, fontsize=12)

            if cd_col_name in df_plot_beta0_clcd_proc_filtered.columns:
                ax2.plot(df_plot_beta0_clcd_proc_filtered['AoA'], df_plot_beta0_clcd_proc_filtered[cd_col_name],
                         linestyle='-', marker='none', markersize=4, color=color_cd_proc,
                         label=f'{cd_col_name} with Stall', alpha=0.7)
            if cd_col_name in df_plot_beta0_clcd_orig_filtered.columns:
                ax2.plot(df_plot_beta0_clcd_orig_filtered['AoA'], df_plot_beta0_clcd_orig_filtered[cd_col_name],
                         linestyle='--', marker='o', markersize=6, color=color_cd_orig,
                         label=f'{cd_col_name}', alpha=0.6)
            ax2.tick_params(axis='y', labelcolor=color_cd_proc)

            # Title and Limits
            plt.title(f'{cl_col_name} and {cd_col_name} vs. AoA (Annular wing) for V = 20 m/s', fontsize=14)
            ax1.set_xlim(0 - EPSILON, 25 + EPSILON) # Set x-axis limits

            # Stall onset line (if applicable within the new range)
            if STALL_AOA_POINT >= 0 and STALL_AOA_POINT <= 25:
                ax1.axvline(STALL_AOA_POINT, color='black', linestyle='--', linewidth=1.2, label=f'Stall Onset ({STALL_AOA_POINT:.1f} deg)')

            # Grids and zero lines
            ax1.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
            ax1.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
            ax1.minorticks_on()
            ax1.axhline(0, color='black', linewidth=1.0) # For CL
            # # ax2.axhline(0, color='darkgray', linewidth=1.0, linestyle=':') # CDtot usually starts > 0

            # Legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            print(f"No data available for {cl_col_name}/{cd_col_name} combined plot in AoA range 0-25 for Beta=0.")
    # --- END of new CL/CD plot section ---
        # --- NEW: Moment Coefficient (CMy) Plot for Beta=0, Stall Modified, AoA 0-25 deg ---
    cmy_col_name = 'CMy'
    if cmy_col_name in df_plot_beta0_mirrored.columns:
        # Filter data for AoA range 0 to 25 from the processed (stalled & mirrored) Beta=0 data
        df_cmy_proc_filtered = df_plot_beta0_mirrored[
            (df_plot_beta0_mirrored['AoA'] >= -EPSILON) & (df_plot_beta0_mirrored['AoA'] <= 25 + EPSILON)
        ].copy()

        # Filter original input data for comparison
        df_cmy_orig_filtered = df_input_beta0_orig_plot[
            (df_input_beta0_orig_plot['AoA'] >= -EPSILON) & (df_input_beta0_orig_plot['AoA'] <= 25 + EPSILON)
        ].copy()
        
        if not df_cmy_proc_filtered.empty or not df_cmy_orig_filtered.empty:
            plt.figure(figsize=(16, 9))

            if cmy_col_name in df_cmy_proc_filtered.columns:
                plt.plot(df_cmy_proc_filtered['AoA'], df_cmy_proc_filtered[cmy_col_name],
                         linestyle='-', marker='none', color='purple',
                         label=f'{cmy_col_name} with Stall', alpha=0.7)
            
            if cmy_col_name in df_cmy_orig_filtered.columns:
                plt.plot(df_cmy_orig_filtered['AoA'], df_cmy_orig_filtered[cmy_col_name],
                         linestyle=':', marker='o', markersize=6, color='orange',
                         label=f'{cmy_col_name}', alpha=0.6)

            plt.xlabel('AoA (degrees)', fontsize=12)
            plt.ylabel(f'{cmy_col_name} (-)', fontsize=12)
            plt.title(f'{cmy_col_name} vs. AoA (Annular Wing) for V = 20 m/s', fontsize=14)
            plt.xlim(0 - EPSILON, 25 + EPSILON) # Set x-axis limits

            if STALL_AOA_POINT >= 0 and STALL_AOA_POINT <= 25:
                plt.axvline(STALL_AOA_POINT, color='black', linestyle='--', linewidth=1.2, label=f'Stall Onset ({STALL_AOA_POINT:.1f} deg)')

            plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
            plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
            plt.minorticks_on()
            plt.axhline(0, color='black', linewidth=1.0) # Zero line for CMy

            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            print(f"No data available for {cmy_col_name} plot in AoA range 0-25 for Beta=0.")
    # --- END of new CMy plot section ---


    # --- NEW: Drag Polar Plot (CL vs CDtot) for Beta=0, Stall Modified, AoA 0-25 deg ---
    if cl_col_name in df_plot_beta0_mirrored.columns and cd_col_name in df_plot_beta0_mirrored.columns:
        # Filter data for AoA range 0 to 25 from the processed (stalled & mirrored) Beta=0 data
        df_drag_polar_filtered = df_plot_beta0_mirrored[
            (df_plot_beta0_mirrored['AoA'] >= -EPSILON) & (df_plot_beta0_mirrored['AoA'] <= 25 + EPSILON)
        ].copy()

        if not df_drag_polar_filtered.empty:
            plt.figure(figsize=(12, 8)) # Adjusted figure size for drag polar

            plt.plot(df_drag_polar_filtered[cd_col_name], df_drag_polar_filtered[cl_col_name],
                     linestyle='-', marker='None', markersize=6, color='navy',
                     label='V = 20 m/s')

            plt.xlabel('CD (-)', fontsize=12)
            plt.ylabel('CL (-)', fontsize=12)
            plt.title('Drag Polar (Annular Wing)', fontsize=14)
            
            plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
            plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
            plt.minorticks_on()
            
            # Optional: Add lines for L/D max if desired, would require calculation
            # Example: Find point of max L/D
            # if not df_drag_polar_filtered.empty and 'L/D' in df_drag_polar_filtered.columns:
            #     df_drag_polar_filtered_ld = df_drag_polar_filtered.dropna(subset=['L/D'])
            #     if not df_drag_polar_filtered_ld.empty:
            #         max_ld_row = df_drag_polar_filtered_ld.loc[df_drag_polar_filtered_ld['L/D'].idxmax()]
            #         plt.plot([0, max_ld_row[cd_col_name]], [0, max_ld_row[cl_col_name]], 
            #                  color='gray', linestyle='--', linewidth=1, label=f'Max L/D ~ {max_ld_row["L/D"]:.2f}')
            #         plt.scatter(max_ld_row[cd_col_name], max_ld_row[cl_col_name], color='gold', s=100, zorder=5, edgecolors='black', label='Point of Max L/D')


            plt.legend(loc='best', fontsize=10)
            plt.tight_layout()
            plt.show()
            plt.close()

    # --- Original loop for individual plots (can be kept or modified) ---
    for coeff_col_name, coeff_label_short in coeffs_to_plot.items():
        # Skip individual CL and CDtot plots if you only want the combined one for Beta=0
        # if coeff_col_name in [cl_col_name, cd_col_name]:
        #     continue

        plt.figure(figsize=(16, 9))

        # Plot original input Beta=0 data (typically positive AoA only for comparison)
        df_input_beta0_orig_plot_filtered_for_individual = df_input_beta0_orig_plot[df_input_beta0_orig_plot['AoA'] >= -EPSILON]
        if coeff_col_name in df_input_beta0_orig_plot_filtered_for_individual.columns:
            plt.plot(df_input_beta0_orig_plot_filtered_for_individual['AoA'], df_input_beta0_orig_plot_filtered_for_individual[coeff_col_name],
                     linestyle=':', marker='o', markersize=6, color='blue', label='Original Input Data (Beta=0, Positive AoA)', alpha=0.6)

        # Plot fully processed Beta=0 data (stalled and AoA-mirrored)
        if coeff_col_name in df_plot_beta0_mirrored.columns:
            plt.plot(df_plot_beta0_mirrored['AoA'], df_plot_beta0_mirrored[coeff_col_name],
                     linestyle='-', marker='.', markersize=4, color='red', label='Processed Data (Beta=0, Stalled & Mirrored AoA)', alpha=0.7)
        else:
            print(f"Warning: Column {coeff_col_name} not found for plotting Beta=0 (processed).")

        plt.xlabel('AoA (degrees)', fontsize=12)
        plt.ylabel(f'{coeff_label_short}', fontsize=12)
        plt.title(f'{coeff_label_short} vs. AoA (Beta=0) - Axisymmetric Mirroring', fontsize=14)

        plt.axvline(STALL_AOA_POINT, color='green', linestyle='--', linewidth=1.2, label=f'Stall Onset ({STALL_AOA_POINT:.1f} deg)')
        plt.axvline(-STALL_AOA_POINT, color='purple', linestyle='--', linewidth=1.2, label=f'Mirrored Stall Onset ({-STALL_AOA_POINT:.1f} deg)')

        plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
        plt.minorticks_on(); plt.axhline(0, color='black', linewidth=1.0); plt.axvline(0, color='black', linewidth=1.0)

        min_aoa_plot_limit, max_aoa_plot_limit = - (CM_PLATEAU_END_LABEL + 5), (CM_PLATEAU_END_LABEL + 5)
        if not df_plot_beta0_mirrored.empty:
            min_aoa_data = df_plot_beta0_mirrored['AoA'].min()
            max_aoa_data = df_plot_beta0_mirrored['AoA'].max()
            padding = max((max_aoa_data - min_aoa_data) * 0.05, 5.0)
            min_aoa_plot_limit = min_aoa_data - padding
            max_aoa_plot_limit = max_aoa_data + padding

        plt.xlim(left=min_aoa_plot_limit, right=max_aoa_plot_limit)

        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.show()
        plt.close()


    # # Optional: Plotting for a non-zero Beta slice to show Beta mirroring effect
    # target_beta_for_plot = 2.0 # Example Beta value from your CSV
    # # ... (rest of your plotting code for non-zero Beta) ...
    # df_plot_beta_target_mirrored = df_fully_mirrored_and_stalled[
    #     np.isclose(df_fully_mirrored_and_stalled['Beta'].fillna(1e9), target_beta_for_plot)
    # ].sort_values(by='AoA')
    
    # df_plot_beta_neg_target_mirrored = df_fully_mirrored_and_stalled[
    #     np.isclose(df_fully_mirrored_and_stalled['Beta'].fillna(1e9), -target_beta_for_plot)
    # ].sort_values(by='AoA')

    # if not df_plot_beta_target_mirrored.empty and not df_plot_beta_neg_target_mirrored.empty:
    #     print(f"\nGenerating plots for Beta = +/-{target_beta_for_plot} deg...")
    #     for coeff_col_name, coeff_label_short in coeffs_to_plot.items():
    #         plt.figure(figsize=(16, 9))
            
    #         if coeff_col_name in df_plot_beta_target_mirrored.columns:
    #             plt.plot(df_plot_beta_target_mirrored['AoA'], df_plot_beta_target_mirrored[coeff_col_name], 
    #                      linestyle='-', marker='.', markersize=4, color='orange', label=f'Processed Data (Beta={target_beta_for_plot:.0f} deg)', alpha=0.7)
            
    #         if coeff_col_name in df_plot_beta_neg_target_mirrored.columns:
    #             plt.plot(df_plot_beta_neg_target_mirrored['AoA'], df_plot_beta_neg_target_mirrored[coeff_col_name], 
    #                      linestyle='--', marker='x', markersize=4, color='magenta', label=f'Processed Data (Beta={-target_beta_for_plot:.0f} deg, Mirrored Beta)', alpha=0.7)

    #         plt.xlabel('AoA (degrees)', fontsize=12)
    #         plt.ylabel(f'{coeff_label_short}', fontsize=12)
    #         plt.title(f'{coeff_label_short} vs. AoA (Beta=+/-{target_beta_for_plot:.0f} deg) - Axisymmetric Mirroring', fontsize=14)
    #         plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7) 
    #         plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
    #         plt.minorticks_on(); plt.axhline(0, color='black', linewidth=1.0); plt.axvline(0, color='black', linewidth=1.0)
    #         plt.legend(loc='best', fontsize=10)
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close() # Added close here
    else:
        print(f"Could not find sufficient data for Beta=+/-{target_beta_for_plot} to generate comparison plots.")