import openvsp_config
import numpy as np
import time
import openvsp as vsp
import os
import math
import json
from plotter import plot_polar_data 
from plotter import read_polar_file 
from stall import simulate_stall_behavior
openvsp_config.LOAD_GRAPHICS = False
openvsp_config.LOAD_FACADE = False
import glob
# Ensure OpenVSP is initialized 

# --- Atmospheric Constants (Standard Sea Level ISA) ---
STD_TEMP_K = 288.15
GAMMA_AIR = 1.4
R_SPECIFIC_AIR = 287.058
STD_PRESSURE_PA = 101325.0
MU_AIR_15C = 1.81e-5 # Pa.s (kg/(m.s))
RHO_AIR_15C_SL = STD_PRESSURE_PA / (R_SPECIFIC_AIR * STD_TEMP_K) # kg/m^3 ~1.225
SPEED_OF_SOUND_A_15C_SL = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * STD_TEMP_K) # m/s ~340.29


def run_prop_analysis(params_file_path):
    print(f"Python: Loading parameters from: {params_file_path}")
    if not os.path.exists(params_file_path):
        print(f"Python: ERROR - Parameters JSON file not found: {params_file_path}")
        return None, None # Or raise an error: sys.exit(f"ERROR: Parameters JSON file not found: {params_file_path}")

    try:
        with open(params_file_path, 'r') as f:
            params = json.load(f)
    except Exception as e:
        print(f"Python: ERROR - Could not read or parse JSON parameters file: {e}")
        return None, None 

    # Extract parameters from the loaded dictionary
    vsp3_file_path = params.get('vsp_file_path')
    effective_output_dir = params.get('output_data_folder')
    analysis_settings = params.get('analysis_settings', {})
    tessellation_settings = analysis_settings.get('tessellation_settings', {})
    geometry_BOR_settings = tessellation_settings.get('BORGeom', {})
    geometry_PROP_settings = tessellation_settings.get('PropGeom', {})
    sweep_settings = params.get('sweep_settings', {})
    plot_settings = params.get('plot_settings', {'do_plot': False}) # Default to no plot if not specified
    vsp3_file_basename = os.path.splitext(os.path.basename(vsp3_file_path))[0]
    if not vsp3_file_path:
        print("Python: ERROR - 'vsp_file_path' not found in parameters JSON.")
        return None, None

    print(f"Python: Cleaning VSPAERO-related files from: {os.path.abspath(effective_output_dir)}")
    if os.path.exists(effective_output_dir):
        files_to_delete = [
            f"{vsp3_file_basename}_VSPGeom.*",   # Catches .vspaero, .tri, .map, .polar, .lod, .history etc.
            f"{vsp3_file_basename}_DegenGeom.*" # Also catch if DegenGeom naming is used
        ]
        for pattern in files_to_delete:
            for f_path in glob.glob(os.path.join(effective_output_dir, pattern)):
                try:
                    print(f"  Deleting: {f_path}")
                    os.remove(f_path)
                except Exception as e:
                    print(f"  Warning: Could not delete {f_path}: {e}")

    print(f"Using ISA Sea Level: Temp={STD_TEMP_K-273.15:.1f}C, Rho={RHO_AIR_15C_SL:.3f}kg/m^3, Mu={MU_AIR_15C:.2e}Pa.s, a={SPEED_OF_SOUND_A_15C_SL:.2f}m/s")
    print("Beginning OpenVSP analysis function (using JSON params).")   

    #vsp.VSPCheckSetup()
    print("OpenVSP version:", vsp.GetVSPVersion())
    print("--> Initializing Geometry")
    vsp.ClearVSPModel()
    vsp.Update()
    print(f"Reading VSP3 file: {vsp3_file_path}")
    vsp.ReadVSPFile(vsp3_file_path)
    vsp.Update()
    print("VSP3 file loaded.\n")
    is_prop_analysis = analysis_settings.get('is_prop_analysis', True)
    method_choice = analysis_settings.get('method_choice', 'vlm') 
    prop_geom_name = analysis_settings.get('prop_geom_name', "Prop") 
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
    tess_u_val = geometry_PROP_settings.get('Tess_U')
    tess_w_val = geometry_PROP_settings.get('Tess_W')
    print(tess_u_val, tess_w_val)
    target_component_name_for_tess = "PropGeom" 

    applied_tess_changes = False
    if tess_u_val is not None or tess_w_val is not None: 
        print("\n--- Applying Specific Tessellation Parameter Changes ---")
        comp_id = vsp.FindGeom(target_component_name_for_tess, 0)
        print(f"Comp_ID: {comp_id}") 

        if comp_id: 
            print(f"Found component '{target_component_name_for_tess}' (ID: {comp_id}) for tessellation update.")
            if tess_u_val is not None:
                try:
                    vsp.SetParmVal(comp_id, "Tess_U", "Shape", float(tess_u_val))
                    print(f"  Set 'Tess_U' for '{target_component_name_for_tess}' to {tess_u_val}")
                    applied_tess_changes = True
                except Exception as e:
                    print(f"  WARNING: Could not set 'Tess_U' for '{target_component_name_for_tess}'. Error: {e}")
            
            if tess_w_val is not None:
                try:
                    vsp.SetParmVal(comp_id, "Tess_W", 
                    "Shape", float(tess_w_val))
                    print(f"  Set 'Tess_W' for '{target_component_name_for_tess}' to {tess_w_val}")
                    applied_tess_changes = True
                except Exception as e:
                    print(f"  WARNING: Could not set 'Tess_W' for '{target_component_name_for_tess}'. Error: {e}")
            
            if applied_tess_changes:
                vsp.Update() 
                print("Specific tessellation changes applied and model updated.")
        else:
            print(f"  WARNING: Component named '{target_component_name_for_tess}' not found in VSP model. Tessellation skipped.")
    elif 'tessellation_settings' in analysis_settings and 'BORGeom' in tessellation_settings:
        print("\n--- Tessellation for 'BORGeom' Skipped: 'Tess_U' or 'Tess_W' not specified or null in JSON. ---")

    # Validate essential reference dimensions
    if b_input is None or c_input is None:
        print("Python: ERROR - 'ref_span_b' or 'ref_chord_c' not found or is null in analysis_settings of JSON.")
        return None, None
    
    vsp.SetParmVal(vsp.FindParm(vsp.FindUnsteadyGroup(0),"RPM","UnsteadyGroup"), rpm)
    vsp.Update()

    print("\n--- Running VSPAEROComputeGeometry ---")
    analysis_compute_geom_name = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(analysis_compute_geom_name)
    analysis_method_geom = list(vsp.GetIntAnalysisInput(analysis_compute_geom_name, "AnalysisMethod"))
    analysis_method_geom[0] = vsp.VORTEX_LATTICE # Fallback for ROTORCRAFT
    vsp.SetIntAnalysisInput(analysis_compute_geom_name, "AnalysisMethod", analysis_method_geom)
    if is_prop_analysis:
        print(f"Propeller analysis detected. Setting ComputeGeometry for propeller.")
        # Set PropID if provided
        if prop_geom_name:
            try:
                vsp.SetStringAnalysisInput(analysis_compute_geom_name, "PropID", (prop_geom_name,), 0)
                print(f"ComputeGeometry: Set PropID to {prop_geom_name}")
            except Exception as e:
                print(f"ComputeGeometry: Warning - Could not set PropID to {prop_geom_name}: {e}")

    print("\n\tExecuting VSPAEROComputeGeometry...")
    compute_geom_success_id = vsp.ExecAnalysis(analysis_compute_geom_name)
    if compute_geom_success_id: # vsp.ExecAnalysis returns a non-zero ID on successful submission
        print(f"✅ VSPAEROComputeGeometry executed successfully (ID: {compute_geom_success_id}).")
    else:
        print(f"❌ VSPAEROComputeGeometry failed. Check VSP console for errors. Aborting further analysis.")
        vsp.ClearVSPModel()
        return None, None
    vsp.Update()
    print("VSPAEROComputeGeometry completed.\n")


    print("--> Setting up VSPAERO")
    analysis_name = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(analysis_name)
    vsp.SetIntAnalysisInput(analysis_name, "AnalysisMethod", [0])
    print(f"VSPAEROSweep: AnalysisMethod set to VLM.")

    vsp.SetIntAnalysisInput(analysis_name, "RotateBladesFlag", (1,), 0) # Enable blade rotation
    print("VSPAEROSweep: RotateBladesFlag set to 1 (True)")


    if prop_geom_name:
        try:
            vsp.SetStringAnalysisInput(analysis_name, "PropID", (prop_geom_name,), 0)
            print(f"VSPAEROSweep: Set PropID to '{prop_geom_name}'")
        except Exception as e:
            print(f"VSPAEROSweep: Warning - Could not set PropID to '{prop_geom_name}': {e}")
    
    # --- Reference Dimensions ---
    s_calculated = b_input * c_input * 3 #times amount of blades
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
    
    # # Mach settings
    # if sweep_mode == 'alpha' or sweep_mode == 'alpha_beta':
    #     vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", (mach_input,), 0)
    #     vsp.SetDoubleAnalysisInput(analysis_name, "MachEnd", (mach_input,), 0)
    #     vsp.SetIntAnalysisInput(analysis_name, "MachNpts", (1,), 0)
    #     print(f"  VSPAERO Mach SINGLE: {mach_input:.4f}")
    # elif sweep_mode == 'velocity':
    #     vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", (mach_start_input,), 0)
    #     vsp.SetDoubleAnalysisInput(analysis_name, "MachEnd", (mach_end_input,), 0)
    #     vsp.SetIntAnalysisInput(analysis_name, "MachNpts", (num_vel_points_input,), 0)
    #     print(f"  VSPAERO Mach SWEEP: {num_vel_points_input} pt(s) from {mach_start_input:.4f} to {mach_end_input:.4f}")

    vsp.SetDoubleAnalysisInput(analysis_name, "Vinf", (v_inf_input_ms,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "Vref", (v_inf_input_ms,), 0)
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
    num_time_steps = analysis_settings.get('num_time_step')
    time_step_size = analysis_settings.get('size_time_step') 
    vsp.SetIntAnalysisInput(analysis_name, "AutoTimeStepFlag", (0,), 0) 
    vsp.SetIntAnalysisInput(analysis_name, "WakeNumIter", (wake_iter_input,),0); print(f"  WakeNumIter: {wake_iter_input}")
    vsp.SetIntAnalysisInput(analysis_name, "Symmetry", (symm_flag_input,),0); print(f"  Symmetry: {symm_flag_input} ({symm_options.get(symm_flag_input, 'Unknown Code')})")
    vsp.SetIntAnalysisInput(analysis_name, "NCPU", (ncpu,),0); print(f"  NCPU: {ncpu}")
    vsp.SetIntAnalysisInput(analysis_name, "NumWakeNodes", (wake_num_nodes_input,), 0); print(f"NumWakeNodes: {wake_num_nodes_input}")
    vsp.SetDoubleAnalysisInput(analysis_name, "TimeStepSize", (time_step_size,), 0)
    vsp.SetIntAnalysisInput(analysis_name, "NumTimeSteps", (num_time_steps,), 0); print(f"NumTimeSteps: {num_time_steps} (TimeStepSize={time_step_size:.4f}s)")

    

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

     
    print("\n\tExecuting VSPAERO...")
    rid = vsp.ExecAnalysis(analysis_name)
    print("VSPAERO Execution COMPLETE")

    # --- Get and Display Results ---
    vsp3_file_basename = os.path.splitext(os.path.basename(vsp3_file_path))[0]
    effective_output_dir = os.path.dirname(vsp3_file_path) # Default if no output_data_folder
    if not effective_output_dir: # If vsp3_file_path was just a filename
        effective_output_dir = "."

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

    rotor_results_path = None  # Initialize to None

    rotor_file_name_vspgeom = f"{vsp3_file_basename}_VSPGeom.rotor.1"
    rotor_file_path_vspgeom = os.path.join(effective_output_dir, rotor_file_name_vspgeom)
    rotor_file_name_degengeom = f"{vsp3_file_basename}_DegenGeom.rotor.1"
    rotor_file_path_degengeom = os.path.join(effective_output_dir, rotor_file_name_degengeom)

    if os.path.exists(rotor_file_path_vspgeom):
        rotor_results_path = rotor_file_path_vspgeom
        print(f"Found .rotor file: {rotor_results_path}")
    elif os.path.exists(rotor_file_path_degengeom):
        rotor_results_path = rotor_file_path_degengeom
        print(f"Found .rotor file: {rotor_results_path}")
    else:
        print(f"Note: No .rotor file found for '{vsp3_file_basename}' in '{os.path.abspath(effective_output_dir)}'")

    if hasattr(vsp, 'ClearVSPModel'): # Check if vsp object and method exist
        vsp.ClearVSPModel()
    else:
        print("Warning: vsp.ClearVSPModel not available.")
        
    print("\nPropeller Analysis function finished.")
    return rotor_results_path



if __name__ == "__main__":
    print("This Python script 'vsp_analysis_propeller.py' provides analysis functions.")
    test_params_file_prop = r'C:\Users\tomsp\PycharmProjects\pythonProject1\DSE\prop_params.json' # CHANGEEEEE
    if os.path.exists(test_params_file_prop):
        print(f"\n--- Running Standalone Python Test (Propeller) with {test_params_file_prop} ---")
        rotorf_p = run_prop_analysis(test_params_file_prop)
        if rotorf_p is not None:
            print(f"Standalone Propeller Test Rotor File: {rotorf_p}")
            print(rotorf_p.head())
            print(f"  Columns available: {rotorf_p.columns.tolist()}")
    else:
        print(f"Test file {test_params_file_prop} not found. Create it based on the example prop_params dict above. Skipping propeller test.")