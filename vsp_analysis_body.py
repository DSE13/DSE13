import openvsp_config
import pandas as pd
import numpy as np
import json 
import os
import sys
from plotter import plot_polar_data # Assuming plotter.py is in the same directory or accessible
from plotter import read_polar_file # Assuming plotter.py with this function
openvsp_config.LOAD_GRAPHICS = False
openvsp_config.LOAD_FACADE = False
import openvsp as vsp
import os
import subprocess # Not used in the provided script, but kept if OpenVSP uses it indirectly
import math

# --- Atmospheric Constants (Standard Sea Level ISA) ---
STD_TEMP_K = 288.15
GAMMA_AIR = 1.4
R_SPECIFIC_AIR = 287.058
STD_PRESSURE_PA = 101325.0
MU_AIR_15C = 1.81e-5 # Pa.s (kg/(m.s))
RHO_AIR_15C_SL = STD_PRESSURE_PA / (R_SPECIFIC_AIR * STD_TEMP_K) # kg/m^3 ~1.225
SPEED_OF_SOUND_A_15C_SL = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * STD_TEMP_K) # m/s ~340.29

def run_openvsp_analysis(params_file_path):
    """
    Performs OpenVSP aerodynamic analysis based on parameters loaded from a JSON file.

    Args:
        params_file_path (str): Full path to the JSON file containing parameters.
    
    Returns:
        tuple: (pandas.DataFrame or None, str or None)
               - DataFrame of the polar data if successful, else None.
               - Path to the rotor results file if applicable and found, else None.
    """

    print(f"Python: Loading parameters from: {params_file_path}")
    if not os.path.exists(params_file_path):
        print(f"Python: ERROR - Parameters JSON file not found: {params_file_path}")
        return None, None # Or raise an error: sys.exit(f"ERROR: Parameters JSON file not found: {params_file_path}")

    try:
        with open(params_file_path, 'r') as f:
            params = json.load(f)
    except Exception as e:
        print(f"Python: ERROR - Could not read or parse JSON parameters file: {e}")
        return None, None # Or raise

    # Extract parameters from the loaded dictionary
    vsp3_file_path = params.get('vsp_file_path')
    effective_output_dir = params.get('output_data_folder')
    # output_data_folder = params.get('output_data_folder', '.') # Example for output
    analysis_settings = params.get('analysis_settings', {})
    sweep_settings = params.get('sweep_settings', {})
    plot_settings = params.get('plot_settings', {'do_plot': False}) # Default to no plot if not specified

    if not vsp3_file_path:
        print("Python: ERROR - 'vsp_file_path' not found in parameters JSON.")
        return None, None

    print(f"Using ISA Sea Level: Temp={STD_TEMP_K-273.15:.1f}C, Rho={RHO_AIR_15C_SL:.3f}kg/m^3, Mu={MU_AIR_15C:.2e}Pa.s, a={SPEED_OF_SOUND_A_15C_SL:.2f}m/s")
    print("Beginning OpenVSP analysis function (using JSON params).")   

    #vsp.VSPCheckSetup()
    print("OpenVSP version:", vsp.GetVSPVersion())
    

    # --- Determine effective_output_dir ---
    vsp3_file_basename = os.path.splitext(os.path.basename(vsp3_file_path))[0]
    
    # Default output dir is where the VSP3 file is
    default_output_dir_based_on_vsp3 = os.path.dirname(vsp3_file_path)
    if not default_output_dir_based_on_vsp3: # If vsp3_file_path was just "model.vsp3"
        default_output_dir_based_on_vsp3 = os.getcwd() # Fallback to current Python script CWD

    effective_output_dir = default_output_dir_based_on_vsp3 # Initialize

    specified_output_folder_from_json = params.get('output_data_folder')
    
    if specified_output_folder_from_json:
        abs_specified_output_folder = os.path.abspath(specified_output_folder_from_json)
        if not os.path.exists(abs_specified_output_folder):
            try:
                os.makedirs(abs_specified_output_folder, exist_ok=True)
                print(f"Python: Created output data folder: {abs_specified_output_folder}")
                effective_output_dir = abs_specified_output_folder
            except OSError as e:
                print(f"Python: WARNING - Could not create specified output_data_folder {abs_specified_output_folder}: {e}. Using default: {effective_output_dir}")
        elif os.path.isdir(abs_specified_output_folder):
            effective_output_dir = abs_specified_output_folder
        else:
            print(f"Python: WARNING - Specified output_data_folder '{abs_specified_output_folder}' exists but is not a directory. Using default: {effective_output_dir}")
    
    print(f"Python: VSPAERO will be run in and write outputs to: {os.path.abspath(effective_output_dir)}")

    print("--> Initializing Geometry")
    vsp.ClearVSPModel()
    vsp.Update()
    print(f"Reading VSP3 file: {vsp3_file_path}")
    vsp.ReadVSPFile(vsp3_file_path)
    vsp.Update()
    print("VSP3 file loaded.\n")
    is_prop_analysis = analysis_settings.get('is_prop_analysis', False)
    # For non-prop, get the method_choice; default to 'panel' if not specified
    method_choice = analysis_settings.get('method_choice') if not is_prop_analysis else None
    prop_geom_name = analysis_settings.get('prop_geom_name', "Prop") # Default if not in JSON
    rpm = analysis_settings.get('rpm')

    b_input = analysis_settings.get('ref_span_b')
    c_input = analysis_settings.get('ref_chord_c')
    xcg_input = analysis_settings.get('xcg')
    wake_iter_input = analysis_settings.get('wake_iterations')
    symm_flag_input = analysis_settings.get('symmetry_flag')
    ncpu = analysis_settings.get('ncpu', os.cpu_count() or 4) # Default if not in JSON
    wake_num_nodes_input = analysis_settings.get('wake_num_nodes')
    wake_type_input = analysis_settings.get('wake_type')
    far_field_dist_factor_input = analysis_settings.get('far_field_dist_factor')
    

    # Validate essential reference dimensions
    if b_input is None or c_input is None:
        print("Python: ERROR - 'ref_span_b' or 'ref_chord_c' not found or is null in analysis_settings of JSON.")
        return None, None

    print("--> Setting up VSPAERO")
    analysis_name = "VSPAEROSweep"
    in_names =  vsp.GetAnalysisInputNames( analysis_name )

    print("Analysis Inputs: ")

    for i in range(int( len(in_names) )):

        print( ( "\t" + in_names[i] + "\n" ) )

    # --- Analysis Type (VLM/Panel) ---
    analysis_method_values = list(vsp.GetIntAnalysisInput(analysis_name, "AnalysisMethod"))

    print("\n--- Setting up for Standard VLM/Panel Sweep Analysis ---")
    if method_choice == 'panel':
        analysis_method_values[0] = vsp.PANEL
        print(f"AnalysisMethod set to PANEL: {analysis_method_values[0]}")
    elif method_choice == 'vlm':
        analysis_method_values[0] = vsp.VORTEX_LATTICE
        print(f"AnalysisMethod set to VORTEX_LATTICE: {analysis_method_values[0]}")
    else:
        print(f"Warning: Invalid analysis_method_non_prop '{method_choice}'. Defaulting to PANEL.")
        analysis_method_values[0] = vsp.PANEL
    vsp.SetIntAnalysisInput(analysis_name, "AnalysisMethod", analysis_method_values)
    vsp.SetIntAnalysisInput(analysis_name, "RotateBladesFlag", (0,), 0) # Always False for non-propeller
    print("RotateBladesFlag set to 0 (False)")

    # --- Reference Dimensions ---
    s_calculated = b_input * c_input
    print(f"\nReference Span (b_input): {b_input:.4f} m (e.g. wingspan)")
    print(f"Reference Chord (c_input): {c_input:.4f} m (e.g. MAC)")
    print(f"Reference Area (S_calculated = b*c): {s_calculated:.4f} m^2 (used as Sref)")

    # --- Sweep Type and Parameters ---
    sweep_mode = sweep_settings['sweep_type'].lower()
    v_inf_input_ms = -1.0; mach_input = -1.0
    alpha_start_deg_input = 0.0; alpha_end_deg_input = 0.0; num_alpha_points_input = 1
    beta_start_deg_input = 0.0; beta_end_deg_input = 0.0; num_beta_points_input = 1
    single_alpha_deg_input = 0.0
    mach_start_input = -1.0; mach_end_input = -1.0; num_vel_points_input = 1
    vel_start_ms_input = -1.0; vel_end_ms_input = -1.0

    if sweep_mode == 'alpha' or sweep_mode == 'alpha_beta':
        input_mode = sweep_settings['alpha_sweep_flight_condition_input_type'].lower()
        if input_mode == 'velocity':
            v_inf_input_ms = sweep_settings['alpha_sweep_flight_velocity_ms']
            mach_input = v_inf_input_ms / SPEED_OF_SOUND_A_15C_SL
            print(f"\n{sweep_mode.capitalize()} Sweep Flight Condition: Vinf={v_inf_input_ms:.2f} m/s, Mach={mach_input:.4f}")
        elif input_mode == 'mach':
            mach_input = sweep_settings['alpha_sweep_flight_mach']
            v_inf_input_ms = mach_input * SPEED_OF_SOUND_A_15C_SL
            print(f"\n{sweep_mode.capitalize()} Sweep Flight Condition: Mach={mach_input:.4f}, Vinf={v_inf_input_ms:.2f} m/s")
        else:
            print(f"Error: Invalid alpha_sweep_flight_condition_input_type '{input_mode}'. Aborting.")
            return None

        num_alpha_points_input = sweep_settings['alpha_npts']
        alpha_start_deg_input = sweep_settings['alpha_start_deg']
        alpha_end_deg_input = sweep_settings.get('alpha_end_deg', alpha_start_deg_input if num_alpha_points_input == 1 else alpha_start_deg_input + 10.0)
        if num_alpha_points_input > 1 and alpha_end_deg_input < alpha_start_deg_input:
            print("Warning: alpha_end_deg < alpha_start_deg. Adjusting alpha_end_deg to alpha_start_deg.")
            alpha_end_deg_input = alpha_start_deg_input
        print(f"Alpha Sweep: {num_alpha_points_input} pt(s) from {alpha_start_deg_input:.2f} to {alpha_end_deg_input:.2f} deg.")

        if sweep_mode == 'alpha_beta':
            num_beta_points_input = sweep_settings.get('beta_npts', 1)
            beta_start_deg_input = sweep_settings.get('beta_start_deg', 0.0)
            beta_end_deg_input = sweep_settings.get('beta_end_deg', beta_start_deg_input if num_beta_points_input == 1 else beta_start_deg_input + 5.0)
            if num_beta_points_input > 1 and beta_end_deg_input < beta_start_deg_input:
                print("Warning: beta_end_deg < beta_start_deg. Adjusting beta_end_deg to beta_start_deg.")
                beta_end_deg_input = beta_start_deg_input
            print(f"Beta Sweep: {num_beta_points_input} pt(s) from {beta_start_deg_input:.2f} to {beta_end_deg_input:.2f} deg.")
        else: # alpha sweep only
            num_beta_points_input = 1
            beta_start_deg_input = 0.0
            beta_end_deg_input = 0.0

    elif sweep_mode == 'velocity':
        single_alpha_deg_input = sweep_settings['velocity_sweep_single_alpha_deg']
        print(f"\nVelocity/Mach Sweep Single Alpha: {single_alpha_deg_input:.2f} deg.")
        alpha_start_deg_input = single_alpha_deg_input
        alpha_end_deg_input = single_alpha_deg_input
        num_alpha_points_input = 1
        
        beta_start_deg_input = 0.0 # Assuming single beta for velocity sweep
        beta_end_deg_input = 0.0
        num_beta_points_input = 1

        num_vel_points_input = sweep_settings['velocity_npts']
        input_vel_mach_mode = sweep_settings['velocity_sweep_range_input_type'].lower()

        if input_vel_mach_mode == 'velocity':
            vel_start_ms_input = sweep_settings['velocity_sweep_start_velocity_ms']
            mach_start_input = vel_start_ms_input / SPEED_OF_SOUND_A_15C_SL
            if num_vel_points_input > 1:
                vel_end_ms_input = sweep_settings['velocity_sweep_end_velocity_ms']
                mach_end_input = vel_end_ms_input / SPEED_OF_SOUND_A_15C_SL
            else:
                vel_end_ms_input = vel_start_ms_input; mach_end_input = mach_start_input
            print(f"Velocity Sweep: {num_vel_points_input} pt(s) from {vel_start_ms_input:.2f} to {vel_end_ms_input:.2f} m/s")
            print(f"  Corresponding Mach: from {mach_start_input:.4f} to {mach_end_input:.4f}")
        elif input_vel_mach_mode == 'mach':
            mach_start_input = sweep_settings['velocity_sweep_start_mach']
            vel_start_ms_input = mach_start_input * SPEED_OF_SOUND_A_15C_SL
            if num_vel_points_input > 1:
                mach_end_input = sweep_settings['velocity_sweep_end_mach']
                vel_end_ms_input = mach_end_input * SPEED_OF_SOUND_A_15C_SL
            else:
                mach_end_input = mach_start_input; vel_end_ms_input = vel_start_ms_input
            print(f"Mach Sweep: {num_vel_points_input} pt(s) from {mach_start_input:.4f} to {mach_end_input:.4f}")
            print(f"  Corresponding Velocity: from {vel_start_ms_input:.2f} to {vel_end_ms_input:.2f} m/s")
        else:
            print(f"Error: Invalid velocity_sweep_range_input_type '{input_vel_mach_mode}'. Aborting.")
            return None
    else:
        print(f"Error: Unknown sweep_type '{sweep_mode}'. Aborting.")
        return None
    
    # --- Reynolds Number ---
    re_cref_start_input = -1.0; re_cref_end_input = -1.0; re_cref_npts_input = 1

    if sweep_mode == 'alpha' or sweep_mode == 'alpha_beta':
        re_mode = sweep_settings['alpha_sweep_reynolds_mode'].lower()
        if re_mode == 'calculated':
            final_reynolds_number = (RHO_AIR_15C_SL * v_inf_input_ms * c_input) / MU_AIR_15C
            print(f"\nUsing calculated Reynolds Number (based on Vinf={v_inf_input_ms:.2f}m/s, Cref={c_input:.3f}m): {final_reynolds_number:.0f}")
        elif re_mode == 'manual':
            final_reynolds_number = sweep_settings['alpha_sweep_manual_reynolds']
            print(f"\nUsing manually input Reynolds Number: {final_reynolds_number:.0f}")
        else:
            print(f"Error: Invalid alpha_sweep_reynolds_mode '{re_mode}'. Aborting.")
            return None
        re_cref_start_input = final_reynolds_number
        re_cref_end_input = final_reynolds_number
        re_cref_npts_input = 1
    elif sweep_mode == 'velocity':
        re_cref_npts_input = num_vel_points_input
        re_mode = sweep_settings['velocity_sweep_reynolds_mode'].lower()
        if re_mode == 'calculated':
            re_cref_start_input = (RHO_AIR_15C_SL * vel_start_ms_input * c_input) / MU_AIR_15C
            re_cref_end_input = (RHO_AIR_15C_SL * vel_end_ms_input * c_input) / MU_AIR_15C if num_vel_points_input > 1 else re_cref_start_input
            print(f"\nUsing calculated Reynolds Number range (based on V_start/end, Cref={c_input:.3f}m).")
        elif re_mode == 'manual':
            re_cref_start_input = sweep_settings['velocity_sweep_manual_reynolds_start']
            if num_vel_points_input > 1:
                re_cref_end_input = sweep_settings['velocity_sweep_manual_reynolds_end']
            else:
                re_cref_end_input = re_cref_start_input
            print(f"\nUsing manually input Reynolds Number range.")
        else:
            print(f"Error: Invalid velocity_sweep_reynolds_mode '{re_mode}'. Aborting.")
            return None
    
    print(f"Reynolds Setup: {re_cref_npts_input} pt(s) from {re_cref_start_input:.0f} to {re_cref_end_input:.0f}")

    # --- Moment Reference Point and Solver Settings ---
    symm_options = {0: "No Symmetry", 1: "XZ Plane (pitch)", 2: "XY Plane (yaw/roll)", 3: "YZ Plane (roll/pitch)"}


    print("\n--- Setting VSPAERO Analysis Parameters ---")
    vsp.SetDoubleAnalysisInput(analysis_name, "Xcg", (xcg_input,), 0); print(f"  VSPAERO Xcg: {xcg_input:.3f}m")

    # Alpha settings
    vsp.SetIntAnalysisInput(analysis_name, "AlphaNpts", (num_alpha_points_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaStart", (alpha_start_deg_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaEnd", (alpha_end_deg_input,), 0)
    print(f"  VSPAERO Alpha: {num_alpha_points_input} pt(s) from {alpha_start_deg_input:.2f} to {alpha_end_deg_input:.2f} deg.")

    # Beta settings
    vsp.SetIntAnalysisInput(analysis_name, "BetaNpts", (num_beta_points_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "BetaStart", (beta_start_deg_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "BetaEnd", (beta_end_deg_input,), 0)
    print(f"  VSPAERO Beta: {num_beta_points_input} pt(s) from {beta_start_deg_input:.2f} to {beta_end_deg_input:.2f} deg.")
    
    # Mach settings
    if sweep_mode == 'alpha' or sweep_mode == 'alpha_beta':
        vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", (mach_input,), 0)
        vsp.SetDoubleAnalysisInput(analysis_name, "MachEnd", (mach_input,), 0)
        vsp.SetIntAnalysisInput(analysis_name, "MachNpts", (1,), 0)
        print(f"  VSPAERO Mach SINGLE: {mach_input:.4f}")
    elif sweep_mode == 'velocity':
        vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", (mach_start_input,), 0)
        vsp.SetDoubleAnalysisInput(analysis_name, "MachEnd", (mach_end_input,), 0)
        vsp.SetIntAnalysisInput(analysis_name, "MachNpts", (num_vel_points_input,), 0)
        print(f"  VSPAERO Mach SWEEP: {num_vel_points_input} pt(s) from {mach_start_input:.4f} to {mach_end_input:.4f}")

    vsp.SetDoubleAnalysisInput(analysis_name, "ReCref", (re_cref_start_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "ReCrefEnd", (re_cref_end_input,), 0)
    vsp.SetIntAnalysisInput(analysis_name, "ReCrefNpts", (re_cref_npts_input,), 0)
    print(f"  VSPAERO ReCref: {re_cref_npts_input} pt(s) from {re_cref_start_input:.0f} to {re_cref_end_input:.0f}")

    vsp.SetDoubleAnalysisInput(analysis_name, "Sref", (s_calculated,), 0); print(f"  Sref: {s_calculated:.6f} m^2")
    vsp.SetDoubleAnalysisInput(analysis_name, "bref", (b_input,), 0); print(f"  bref: {b_input:.4f} m")
    vsp.SetDoubleAnalysisInput(analysis_name, "cref", (c_input,), 0); print(f"  cref: {c_input:.4f} m")

    # Wake Type (FixedWakeFlag)
    if wake_type_input is not None:
        try:
            vsp.SetIntAnalysisInput(analysis_name, "FixedWakeFlag", (int(wake_type_input),), 0)
            wake_type_desc = "Rigid" if int(wake_type_input) == 0 else ("Streamline/Relaxed" if int(wake_type_input) == 1 else "Unknown")
            print(f"  WakeType (FixedWakeFlag) set to: {int(wake_type_input)} ({wake_type_desc})")
            if int(wake_type_input) == 0 and wake_iter_input > 0:
                print(f"  Note: WakeType is Rigid (0), but WakeIter is {wake_iter_input}. WakeIter typically applies to Streamline/Relaxed WakeType (1). VSPAERO might ignore WakeIter.")
            if int(wake_type_input) == 1 and wake_iter_input == 0:
                 print(f"  Note: WakeType is Streamline/Relaxed (1), but WakeIter is 0. Consider WakeIter > 0 for relaxation benefits.")
        except Exception as e:
            print(f"  Warning: Could not set WakeType (FixedWakeFlag). Error: {e}. Using VSPAERO default.")
    else:
        print("  WakeType (FixedWakeFlag): Using VSPAERO default.")
        if wake_iter_input > 0:
            print("    (Implied relaxed wake behavior due to WakeIter > 0. Ensure VSPAERO's default FixedWakeFlag aligns or set 'wake_type' explicitly.)")
        else:
            print("    (Implied rigid wake behavior due to WakeIter = 0. Ensure VSPAERO's default FixedWakeFlag aligns or set 'wake_type' explicitly.)")
    
    vsp.SetIntAnalysisInput(analysis_name, "WakeNumIter", (wake_iter_input,),0); print(f"  WakeNumIter: {wake_iter_input}")
    vsp.SetIntAnalysisInput(analysis_name, "Symmetry", (symm_flag_input,),0); print(f"  Symmetry: {symm_flag_input} ({symm_options.get(symm_flag_input, 'Unknown Code')})")
    vsp.SetIntAnalysisInput(analysis_name, "NCPU", (ncpu,),0); print(f"  NCPU: {ncpu}")
    vsp.SetIntAnalysisInput(analysis_name, "NumWakeNodes", (wake_num_nodes_input,), 0); print(f"NumWakeNodes: {wake_num_nodes_input}")

    

    # Far Field Distance
    if far_field_dist_factor_input is not None:
        try:
            actual_far_distance = float(far_field_dist_factor_input) * b_input # Based on ref_span_b
            vsp.SetIntAnalysisInput(analysis_name, "FarDistToggle",  (1,), 0) # Enable manual FarDist
            vsp.SetDoubleAnalysisInput(analysis_name, "FarDist", (float(actual_far_distance),), 0)
            print(f"  FarField Distance set to: {float(actual_far_distance):.2f} m ({far_field_dist_factor_input} * ref_span_b)")
        except Exception as e:
            print(f"  Warning: Could not set FarField Distance. Error: {e}. Using VSPAERO default.")
    else:
        print("  FarField Distance: Using VSPAERO default (FarDistToggle=0).")
    

    print("\nFinal Analysis Inputs (check VSPAERO settings after this if GUI is open):")
    vsp.PrintAnalysisInputs(analysis_name)
    
    # --- Execute VSPAERO in the correct directory ---
    print(f"Python: Current working directory before VSPAERO: {os.getcwd()}")
    print(f"Python: Changing CWD to VSPAERO output directory: {os.path.abspath(effective_output_dir)}")
    original_cwd = os.getcwd()
    polar_df = None # Initialize
    
    try:
        os.chdir(effective_output_dir)
        print(f"Python: CWD is now: {os.getcwd()}")
        print("\n\tExecuting VSPAERO...")
        rid = vsp.ExecAnalysis(analysis_name) # vsp.ExecAnalysis uses the analysis_name to find the settings.
                                              # It generates a DegenGeom and a .vspaero file (e.g., YourModel_DegenGeom.vspaero)
                                              # The argument to vspaero.exe will be "YourModel_DegenGeom"
        print("VSPAERO Execution COMPLETE")
    except Exception as e:
        print(f"Python: ERROR during VSPAERO execution or CWD change: {e}")
        # No polar_df will be available
    finally:
        os.chdir(original_cwd)
        print(f"Python: Restored CWD to: {os.getcwd()}")

    print(f"Python: Expecting/Searching for result files in directory: {os.path.abspath(effective_output_dir)}")

    # --- Get and Display Results ---
    vsp3_file_basename = os.path.splitext(os.path.basename(vsp3_file_path))[0]
    effective_output_dir = os.path.dirname(vsp3_file_path) # Default if no output_data_folder
    if not effective_output_dir: # If vsp3_file_path was just a filename
        effective_output_dir = os.getcwd()

    specified_output_folder = params.get('output_data_folder')
    if specified_output_folder:
        if not os.path.exists(specified_output_folder):
            try:
                os.makedirs(specified_output_folder)
                print(f"Python: Created output data folder: {specified_output_folder}")
            except OSError as e:
                print(f"Python: WARNING - Could not create output_data_folder {specified_output_folder}: {e}. Using default.")
                # Fall back or handle error
        if os.path.isdir(specified_output_folder): # Check if it's a directory after attempting creation
             effective_output_dir = specified_output_folder


    print(f"Python: Expecting/Saving result files in directory: {os.path.abspath(effective_output_dir)}")

    polar_file_name_vspgeom = f"{vsp3_file_basename}_VSPGeom.polar"
    polar_file_path_vspgeom = os.path.join(effective_output_dir, polar_file_name_vspgeom)
    polar_file_name_degengeom = f"{vsp3_file_basename}_DegenGeom.polar" 
    polar_file_path_degengeom = os.path.join(effective_output_dir, polar_file_name_degengeom)
    
    polar_df = None
    polar_file_processed = ""
    csv_output_path = None

    file_to_try_reading = None
    if os.path.exists(polar_file_path_vspgeom):
        file_to_try_reading = polar_file_path_vspgeom
        polar_file_processed = polar_file_name_vspgeom
    elif os.path.exists(polar_file_path_degengeom): 
        print(f"Note: VSPGeom polar file ('{polar_file_name_vspgeom}') not found. Trying DegenGeom polar: '{polar_file_name_degengeom}'")
        file_to_try_reading = polar_file_path_degengeom
        polar_file_processed = polar_file_name_degengeom
    else:
        print(f"Error: No .polar file (neither {polar_file_name_vspgeom} nor {polar_file_name_degengeom}) found for '{vsp3_file_basename}' in '{os.path.abspath(effective_output_dir)}'")

    if file_to_try_reading:
        polar_df = read_polar_file(file_to_try_reading)
        if polar_df is not None and not polar_df.empty:
            print(f"\nSuccessfully read polar data from '{polar_file_processed}'.")
            csv_file_name = f"{os.path.splitext(polar_file_processed)[0]}.csv" # e.g., NACA0012_1_VSPGeom.csv
            csv_output_path = os.path.join(effective_output_dir, csv_file_name)
            write_polar_to_csv(polar_df, csv_output_path)

            if plot_settings and plot_settings.get('do_plot', False):
                x_col = plot_settings.get('x_axis_col')
                y_col = plot_settings.get('y_axis_col')
                title_suffix = plot_settings.get('plot_title_suffix', "")
                plot_title = f"{os.path.basename(polar_file_processed)} {title_suffix}".strip()

                if x_col and y_col and x_col in polar_df.columns and y_col in polar_df.columns:
                    print(f"Plotting '{y_col}' vs '{x_col}' for {plot_title}")
                    plot_polar_data(polar_df, x_col, y_col, plot_title)
                else:
                    print("Plotting skipped: x_axis_col or y_axis_col missing in plot_settings or not in DataFrame.")
                    print(f"  Requested X: '{x_col}', Requested Y: '{y_col}'")
                    print(f"  Available columns: {polar_df.columns.tolist()}")
        elif polar_df is not None and polar_df.empty:
             print(f"Warning: Polar file '{file_to_try_reading}' was read but is empty or parsed incorrectly.")
        else: # polar_df is None
             print(f"Warning: Failed to read or parse polar file '{file_to_try_reading}' using read_polar_file function.")

    vsp.ClearVSPModel() # Clean up
    print("\nAnalysis function finished.")
    return polar_df, csv_output_path



def write_polar_to_csv(polar_df, csv_file_path):
    """
    Writes a pandas DataFrame (from a .polar file) to a CSV file.

    Args:
        polar_df (pd.DataFrame): DataFrame containing the polar data.
        csv_file_path (str): Full path to the output CSV file.
    """
    if polar_df is not None and not polar_df.empty:
        try:
            polar_df.to_csv(csv_file_path, index=False) # index=False to not write pandas index
            print(f"Python: Successfully wrote polar data to CSV: {csv_file_path}")
            return True
        except Exception as e:
            print(f"Python: ERROR - Could not write polar data to CSV '{csv_file_path}': {e}")
            return False
    else:
        print(f"Python: Skipped writing CSV for '{csv_file_path}' as DataFrame is None or empty.")
        return False
    



if __name__ == "__main__":
    print("This Python script 'vsp_analysis_body.py' now expects a JSON file path as an argument to run_openvsp_analysis.")
    test_params_file = 'NACA0012_1_params.json'
    if os.path.exists(test_params_file):
        print(f"\n--- Running Standalone Python Test with {test_params_file} ---")
        df, rf = run_openvsp_analysis(test_params_file)
        if df is not None:
            print("\nStandalone Test Polar Data (first 5 rows):")
            print(df.head())
        if rf:
            print(f"Standalone Test Rotor File: {rf}")