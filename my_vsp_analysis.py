import openvsp_config
import pandas as pd
import numpy as np
from plotter import plot_polar_data
from plotter import read_polar_file
openvsp_config.LOAD_GRAPHICS = False
openvsp_config.LOAD_FACADE = False
import openvsp as vsp
import os
import subprocess
import math







# --- Atmospheric Constants (Standard Sea Level ISA as an example) ---
STD_TEMP_K = 288.15; GAMMA_AIR = 1.4; R_SPECIFIC_AIR = 287.058
STD_PRESSURE_PA = 101325.0; MU_AIR_15C = 1.81e-5 # Pa.s (kg/(m.s))
RHO_AIR_15C_SL = STD_PRESSURE_PA / (R_SPECIFIC_AIR * STD_TEMP_K) # kg/m^3 ~1.225
SPEED_OF_SOUND_A_15C_SL = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * STD_TEMP_K) # m/s ~340.29
print(f"Using ISA Sea Level: Temp={STD_TEMP_K-273.15:.1f}C, Rho={RHO_AIR_15C_SL:.3f}kg/m^3, Mu={MU_AIR_15C:.2e}Pa.s, a={SPEED_OF_SOUND_A_15C_SL:.2f}m/s")
print("Beginning analysis")

print("--> Generating Geometries")
print("")
vsp.VSPCheckSetup()
print("OpenVSP version:", vsp.GetVSPVersion())








# --- USER INPUT FOR VSP3 FILE ---
vsp3_file_path_input = ""
while not vsp3_file_path_input or not os.path.exists(vsp3_file_path_input):
    vsp3_file_path_input = input("Enter the full path to your .vsp3 file: ").strip()
    #C:\Users\tomsp\Downloads\OpenVSP-3.43.0-win64-Python3.11\OpenVSP-3.43.0-win64\......vsp3
    if not vsp3_file_path_input:
        print("No path entered. Please provide a valid .vsp3 file path.")
    elif not os.path.exists(vsp3_file_path_input):
        print(f"Error: File not found at '{vsp3_file_path_input}'. Please check the path.")
# --- END VSP3 FILE INPUT ---




vsp.ClearVSPModel()
vsp.Update()
print(f"Reading VSP3 file: {vsp3_file_path_input}")
vsp.ReadVSPFile(vsp3_file_path_input)
vsp.Update()

print("VSP3 file loaded.")

print("COMPLETE\n")

print("--> Computing VSPAERO")
print("")

# Set up VSPAero Sweep analysis
analysis_name = "VSPAEROSweep"
print(f"Setting up analysis: {analysis_name}")

# Set default inputs
vsp.SetAnalysisInputDefaults(analysis_name)





# --- User Choice for Analysis Type ---
analysis_type_choice = input("Perform propeller/rotor analysis? (yes/no, default: no): ").strip().lower()
is_prop_analysis = (analysis_type_choice == 'yes' or analysis_type_choice == 'y')

if is_prop_analysis:
    print("\n--- Setting up for Propeller/Rotor Analysis ---")
    # 1. Set AnalysisMethod to ROTORCRAFT
    analysis_method_inputs = list(vsp.GetIntAnalysisInput(analysis_name, "AnalysisMethod"))
    analysis_method_inputs[0] = 4 # ROTORCRAFT
    vsp.SetIntAnalysisInput(analysis_name, "AnalysisMethod", analysis_method_inputs)
    print(f"AnalysisMethod set to ROTORCRAFT: {analysis_method_inputs[0]}")

    # 2. Enable Blade Rotation
    vsp.SetIntAnalysisInput(analysis_name, "RotateBladesFlag", (1,), 0)
    print("RotateBladesFlag set to 1 (True)")

    default_prop_geom_name = "Prop"
    prop_geom_name_input = input(f"Enter the Geom Name of the propeller/rotor component (e.g., Prop, RotorBlade, default: {default_prop_geom_name}): ").strip()
    if not prop_geom_name_input:
        prop_geom_name_input = default_prop_geom_name

    rotor_id_set = False
    possible_rotor_id_names = ["RotorID", "WingID", "PropID"]
    for rotor_id_name in possible_rotor_id_names:
        try:
            vsp.SetStringAnalysisInput(analysis_name, rotor_id_name, (prop_geom_name_input,), 0)
            print(f"Rotor component ID ('{rotor_id_name}') set to: {prop_geom_name_input}")
            rotor_id_set = True
            break
        except Exception:
            print(f"Note: Could not set '{rotor_id_name}'. Trying next...")
            pass
    if not rotor_id_set:
        print(f"Warning: Could not automatically set the rotor component ID using common names for '{prop_geom_name_input}'.")
        print("  Please check your VSPAERO documentation for the correct '...ID' input name for propeller/rotor components.")

    default_rpm = 2500.0
    try:
        rpm_input_str = input(f"Enter RPM for the propeller/rotor (default: {default_rpm}): ").strip()
        rpm_input = float(rpm_input_str if rpm_input_str else str(default_rpm))
    except ValueError:
        print(f"Invalid RPM input, using default: {default_rpm}")
        rpm_input = default_rpm

    # Similar to RotorID, RPM input name can vary.
    rpm_set = False
    possible_rpm_names = ["RPM", "RotorRPM", "PropRPM"]
    for rpm_name in possible_rpm_names:
        try:
            vsp.SetDoubleAnalysisInput(analysis_name, rpm_name, (rpm_input,), 0)
            print(f"RPM ('{rpm_name}') set to: {rpm_input}")
            rpm_set = True
            break
        except Exception:
            pass
    if not rpm_set:
        print(f"Warning: Could not automatically set RPM using common names.")
        print("  Please check VSPAERO documentation for the correct RPM input name.")

else:
    print("\n--- Setting up for Standard VLM/Panel Sweep Analysis ---")
    # Set to Panel Method by default for general cases, VLM can also be an option
    analysis_method_input = list(vsp.GetIntAnalysisInput(analysis_name, "AnalysisMethod"))
    # Allow user to choose between PANEL (0) and VLM (1) for non-prop
    method_choice = ""
    while method_choice not in ['p', 'v']:
        method_choice = input("Use (p)anel method or (v)ortex lattice method? [p/v, default: p]: ").strip().lower()
        if not method_choice: method_choice = 'p'

    if method_choice == 'p':
        analysis_method_input[0] = vsp.PANEL # Typically 0
        print(f"AnalysisMethod set to PANEL: {analysis_method_input[0]}")
    else: # 'v'
        analysis_method_input[0] = vsp.VORTEX_LATTICE # Typically 1
        print(f"AnalysisMethod set to VORTEX_LATTICE: {analysis_method_input[0]}")
    vsp.SetIntAnalysisInput(analysis_name, "AnalysisMethod", analysis_method_input )

    vsp.SetIntAnalysisInput(analysis_name, "RotateBladesFlag", (0,), 0)
    print("RotateBladesFlag set to 0 (False for non-propeller analysis)")






# --- USER INPUTS FOR REFERENCE DIMENSIONS ---
print("\n--- Input Reference Dimensions ---")
b_input, c_input = -1.0, -1.0
while b_input <= 0:
    try: b_input = float(input("Enter reference span 'b' (e.g., wing span) in meters: ").strip());
    except ValueError: print("Invalid input.")
    if b_input <= 0: print("Span 'b' must be positive.")
print(f"  Reference Span (b_input): {b_input:.4f} m")
while c_input <= 0:
    try: c_input = float(input("Enter reference chord 'c' (e.g., MAC) in meters: ").strip());
    except ValueError: print("Invalid input.")
    if c_input <= 0: print("Chord 'c' must be positive.")
print(f"  Reference Chord (c_input): {c_input:.4f} m (used for Re calc)")
s_calculated = b_input * c_input # Simple rectangular area, user might adjust Sref later if needed
print(f"  Reference Area (S_calculated = b*c): {s_calculated:.4f} m^2")
# --- END REFERENCE DIMENSIONS ---


# --- CHOOSE SWEEP TYPE ---
print("\n--- Choose Sweep Type ---")
sweep_mode = ""
while sweep_mode not in ['a', 'v']:
    sweep_mode = input("Sweep (a)lpha or (v)elocity/Mach? [a/v, default: a]: ").strip().lower()
    if not sweep_mode: sweep_mode = 'a'


# Initialize sweep parameters - will be set based on sweep_mode
# For Alpha Sweep
v_inf_input_ms = -1.0
mach_input = -1.0 # Single Mach for alpha sweep
num_alpha_points_input = 0
alpha_start_deg_input = 0.0
alpha_end_deg_input = 0.0

# For Velocity/Mach Sweep
single_alpha_deg_input = 0.0 # Single Alpha for velocity sweep
num_vel_points_input = 0
mach_start_input = -1.0
mach_end_input = -1.0
# Derived velocity values if user inputs Mach for velocity sweep
vel_start_ms_input = -1.0
vel_end_ms_input = -1.0


if sweep_mode == 'a':
    print("\n--- Input Flight Condition for Alpha Sweep (Single Velocity/Mach) ---")
    input_mode = ""
    while input_mode not in ['v', 'm']:
        input_mode = input("Specify flight condition by (v)elocity or (m)ach number? [v/m]: ").strip().lower()

    if input_mode == 'v':
        default_velocity_ms = 15.0
        while v_inf_input_ms <= 0:
            try:
                v_str = input(f"Enter freestream velocity Vinf (m/s) (default: {default_velocity_ms}): ").strip()
                v_inf_input_ms = float(v_str if v_str else str(default_velocity_ms))
                if v_inf_input_ms <= 0: print("Velocity must be positive.")
            except ValueError: print("Invalid velocity input.")
        mach_input = v_inf_input_ms / SPEED_OF_SOUND_A_15C_SL
        print(f"  Input Velocity (Vinf): {v_inf_input_ms:.2f} m/s"); print(f"  Calculated Mach Number: {mach_input:.4f}")
    elif input_mode == 'm':
        default_mach_val = 0.13
        while mach_input <= 0:
            try:
                m_str = input(f"Enter Mach number (default: {default_mach_val}): ").strip()
                mach_input = float(m_str if m_str else str(default_mach_val))
                if mach_input <= 0: print("Mach number must be positive.")
            except ValueError: print("Invalid Mach number input.")
        v_inf_input_ms = mach_input * SPEED_OF_SOUND_A_15C_SL
        print(f"  Input Mach Number: {mach_input:.4f}"); print(f"  Calculated Velocity (Vinf): {v_inf_input_ms:.2f} m/s")

    print("\n--- Input Angle of Attack (Alpha) Sweep Parameters ---")
    default_alpha_npts = 1
    while num_alpha_points_input <= 0:
        try:
            npts_str = input(f"Enter number of Alpha points (AlphaNpts) (integer > 0, default: {default_alpha_npts}): ").strip()
            num_alpha_points_input = int(npts_str if npts_str else str(default_alpha_npts))
            if num_alpha_points_input <= 0: print("Number of Alpha points must be a positive integer.")
        except ValueError: print("Invalid input. Please enter an integer.")

    default_alpha_start = 0.0
    try:
        start_str = input(f"Enter Alpha Start (degrees) (default: {default_alpha_start}): ").strip()
        alpha_start_deg_input = float(start_str if start_str else str(default_alpha_start))
    except ValueError:
        print(f"Invalid input for Alpha Start, using default: {default_alpha_start}")
        alpha_start_deg_input = default_alpha_start

    if num_alpha_points_input > 1:
        default_alpha_end = 10.0
        while True:
            try:
                end_str = input(f"Enter Alpha End (degrees) (default: {default_alpha_end}): ").strip()
                alpha_end_deg_input = float(end_str if end_str else str(default_alpha_end))
                if alpha_end_deg_input < alpha_start_deg_input:
                    print("Alpha End cannot be less than Alpha Start for a sweep. Please re-enter.")
                else: break
            except ValueError: print(f"Invalid input for Alpha End, please enter a number.")
    else:
        alpha_end_deg_input = alpha_start_deg_input
    print(f"  Alpha Sweep: {num_alpha_points_input} point(s) from {alpha_start_deg_input:.2f} to {alpha_end_deg_input:.2f} deg.")

elif sweep_mode == 'v':
    print("\n--- Input Single Angle of Attack (Alpha) for Velocity/Mach Sweep ---")
    default_single_alpha = 0.0
    try:
        alpha_str = input(f"Enter single Alpha value (degrees) (default: {default_single_alpha}): ").strip()
        single_alpha_deg_input = float(alpha_str if alpha_str else str(default_single_alpha))
    except ValueError:
        print(f"Invalid input for Alpha, using default: {default_single_alpha}")
        single_alpha_deg_input = default_single_alpha
    print(f"  Single Alpha for sweep: {single_alpha_deg_input:.2f} deg.")

    print("\n--- Input Velocity/Mach Sweep Parameters ---")
    default_vel_npts = 5
    while num_vel_points_input <= 0:
        try:
            npts_str = input(f"Enter number of Velocity/Mach points (Npts) (integer > 0, default: {default_vel_npts}): ").strip()
            num_vel_points_input = int(npts_str if npts_str else str(default_vel_npts))
            if num_vel_points_input <= 0: print("Number of points must be a positive integer.")
        except ValueError: print("Invalid input. Please enter an integer.")

    input_vel_mach_mode = ""
    while input_vel_mach_mode not in ['v', 'm']:
        input_vel_mach_mode = input("Specify sweep range by (v)elocity or (m)ach number? [v/m]: ").strip().lower()

    if input_vel_mach_mode == 'v':
        default_vel_start_ms = 10.0
        while vel_start_ms_input <= 0:
            try:
                v_start_str = input(f"Enter Start Freestream Velocity (m/s) (default: {default_vel_start_ms}): ").strip()
                vel_start_ms_input = float(v_start_str if v_start_str else str(default_vel_start_ms))
                if vel_start_ms_input <= 0: print("Start velocity must be positive.")
            except ValueError: print("Invalid start velocity input.")
        mach_start_input = vel_start_ms_input / SPEED_OF_SOUND_A_15C_SL

        if num_vel_points_input > 1:
            default_vel_end_ms = 50.0
            vel_end_ms_input = -1.0 # ensure it's reset
            while True:
                try:
                    v_end_str = input(f"Enter End Freestream Velocity (m/s) (default: {default_vel_end_ms}): ").strip()
                    vel_end_ms_input = float(v_end_str if v_end_str else str(default_vel_end_ms))
                    if vel_end_ms_input <= 0: print("End velocity must be positive.")
                    elif vel_end_ms_input < vel_start_ms_input: print("End velocity cannot be less than Start velocity.")
                    else: break
                except ValueError: print("Invalid end velocity input.")
            mach_end_input = vel_end_ms_input / SPEED_OF_SOUND_A_15C_SL
        else:
            vel_end_ms_input = vel_start_ms_input
            mach_end_input = mach_start_input
        print(f"  Velocity Sweep: {num_vel_points_input} point(s) from {vel_start_ms_input:.2f} m/s to {vel_end_ms_input:.2f} m/s")
        print(f"  Corresponding Mach Sweep: from {mach_start_input:.4f} to {mach_end_input:.4f}")

    elif input_vel_mach_mode == 'm':
        default_mach_start_val = 0.1
        while mach_start_input <= 0:
            try:
                m_start_str = input(f"Enter Start Mach number (default: {default_mach_start_val}): ").strip()
                mach_start_input = float(m_start_str if m_start_str else str(default_mach_start_val))
                if mach_start_input <= 0: print("Start Mach number must be positive.")
            except ValueError: print("Invalid start Mach number input.")
        vel_start_ms_input = mach_start_input * SPEED_OF_SOUND_A_15C_SL # Store for Re calc

        if num_vel_points_input > 1:
            default_mach_end_val = 0.3
            mach_end_input = -1.0 # ensure it's reset
            while True:
                try:
                    m_end_str = input(f"Enter End Mach number (default: {default_mach_end_val}): ").strip()
                    mach_end_input = float(m_end_str if m_end_str else str(default_mach_end_val))
                    if mach_end_input <= 0: print("End Mach number must be positive.")
                    elif mach_end_input < mach_start_input: print("End Mach cannot be less than Start Mach.")
                    else: break
                except ValueError: print("Invalid end Mach number input.")
            vel_end_ms_input = mach_end_input * SPEED_OF_SOUND_A_15C_SL # Store for Re calc
        else:
            mach_end_input = mach_start_input
            vel_end_ms_input = vel_start_ms_input
        print(f"  Mach Sweep: {num_vel_points_input} point(s) from {mach_start_input:.4f} to {mach_end_input:.4f}")
        print(f"  Corresponding Velocity Sweep: from {vel_start_ms_input:.2f} m/s to {vel_end_ms_input:.2f} m/s")


# --- REYNOLDS NUMBER INPUT ---
print("\n--- Input Reynolds Number ---")
re_cref_start_input = -1.0
re_cref_end_input = -1.0
re_cref_npts_input = 1
final_reynolds_number = -1.0 # Used for single Re case with alpha sweep

if sweep_mode == 'a': # Alpha sweep, single Re value
    if v_inf_input_ms > 0 and c_input > 0:
        calculated_re_suggestion = (RHO_AIR_15C_SL * v_inf_input_ms * c_input) / MU_AIR_15C
        print(f"  Suggested Reynolds Number (based on Vinf={v_inf_input_ms:.2f} m/s and c_input={c_input:.4f} m): {calculated_re_suggestion:.0f}")
        
        re_mode = ""
        while re_mode not in ['c', 'm']:
            re_mode = input("Use (c)alculated Re or input (m)anually? [c/m, default: c]: ").strip().lower()
            if not re_mode: re_mode = 'c'

        if re_mode == 'c':
            final_reynolds_number = calculated_re_suggestion
            print(f"  Using calculated Reynolds Number: {final_reynolds_number:.0f}")
        elif re_mode == 'm':
            while final_reynolds_number <= 0:
                try:
                    re_manual_str = input("Enter Reynolds number (ReCref) manually: ").strip()
                    final_reynolds_number = float(re_manual_str)
                    if final_reynolds_number <= 0: print("Reynolds number must be positive.")
                except ValueError: print("Invalid Reynolds number input.")
            print(f"  Using manually input Reynolds Number: {final_reynolds_number:.0f}")
    else:
        print("  Warning: Cannot calculate or suggest Reynolds number (Vinf or c_input invalid).")
        while final_reynolds_number <= 0:
            print("  Please input Reynolds number manually as suggestion failed.")
            try:
                re_manual_str = input("Enter Reynolds number (ReCref) manually: ").strip()
                final_reynolds_number = float(re_manual_str)
                if final_reynolds_number <= 0: print("Reynolds number must be positive.")
            except ValueError: print("Invalid Reynolds number input.")
    
    re_cref_start_input = final_reynolds_number
    re_cref_end_input = final_reynolds_number
    re_cref_npts_input = 1
    print(f"  Reynolds Number for Alpha Sweep (single value): {re_cref_start_input:.0f}")

elif sweep_mode == 'v': # Velocity/Mach sweep, potentially swept Re
    re_cref_npts_input = num_vel_points_input # Re sweep matches velocity sweep points

    # vel_start_ms_input and vel_end_ms_input are defined from velocity/mach sweep input
    can_calculate_re_range = (vel_start_ms_input > 0 and c_input > 0 and (num_vel_points_input == 1 or vel_end_ms_input > 0))

    if can_calculate_re_range:
        calculated_re_start_suggestion = (RHO_AIR_15C_SL * vel_start_ms_input * c_input) / MU_AIR_15C
        calculated_re_end_suggestion = (RHO_AIR_15C_SL * vel_end_ms_input * c_input) / MU_AIR_15C if num_vel_points_input > 1 else calculated_re_start_suggestion
        
        if num_vel_points_input > 1:
            print(f"  Suggested Reynolds Number Range (based on Vinf sweep and c_input={c_input:.4f} m):")
            print(f"    Re_start: {calculated_re_start_suggestion:.0f} (at Vinf_start={vel_start_ms_input:.2f} m/s)")
            print(f"    Re_end:   {calculated_re_end_suggestion:.0f} (at Vinf_end={vel_end_ms_input:.2f} m/s)")
        else:
            print(f"  Suggested Reynolds Number (based on Vinf={vel_start_ms_input:.2f} m/s and c_input={c_input:.4f} m): {calculated_re_start_suggestion:.0f}")

        re_mode = ""
        while re_mode not in ['c', 'm']:
            re_mode = input("Use (c)alculated Re range/value or input (m)anually? [c/m, default: c]: ").strip().lower()
            if not re_mode: re_mode = 'c'

        if re_mode == 'c':
            re_cref_start_input = calculated_re_start_suggestion
            re_cref_end_input = calculated_re_end_suggestion
            print(f"  Using calculated Reynolds Number range/value.")
        elif re_mode == 'm':
            while re_cref_start_input <= 0:
                try:
                    re_manual_str = input("Enter Start Reynolds number (ReCrefStart) manually: ").strip()
                    re_cref_start_input = float(re_manual_str)
                    if re_cref_start_input <= 0: print("Reynolds number must be positive.")
                except ValueError: print("Invalid Reynolds number input.")
            
            if num_vel_points_input > 1:
                re_cref_end_input = -1.0
                while re_cref_end_input <= 0:
                    try:
                        re_manual_str_end = input("Enter End Reynolds number (ReCrefEnd) manually: ").strip()
                        re_cref_end_input = float(re_manual_str_end)
                        if re_cref_end_input <= 0: print("Reynolds number must be positive.")
                    except ValueError: print("Invalid Reynolds number input.")
            else:
                re_cref_end_input = re_cref_start_input
            print(f"  Using manually input Reynolds Number range/value.")
    else:
        print("  Warning: Cannot calculate or suggest Reynolds number range (Vinf sweep or c_input invalid). Please input manually.")
        while re_cref_start_input <= 0:
            try:
                re_manual_str = input("Enter Start Reynolds number (ReCrefStart) manually: ").strip()
                re_cref_start_input = float(re_manual_str)
                if re_cref_start_input <= 0: print("Reynolds number must be positive.")
            except ValueError: print("Invalid Reynolds number input.")
        
        if num_vel_points_input > 1:
            re_cref_end_input = -1.0
            while re_cref_end_input <= 0:
                try:
                    re_manual_str_end = input("Enter End Reynolds number (ReCrefEnd) manually: ").strip()
                    re_cref_end_input = float(re_manual_str_end)
                    if re_cref_end_input <= 0: print("Reynolds number must be positive.")
                except ValueError: print("Invalid Reynolds number input.")
        else:
            re_cref_end_input = re_cref_start_input
    print(f"  Reynolds Setup for Velocity Sweep: {re_cref_npts_input} pt(s) from {re_cref_start_input:.0f} to {re_cref_end_input:.0f}")


# Get Xcg (Xref)
print("\n--- Input Moment Reference Point ---") # Moved slightly, contextually better here
default_xcg = 0.0
xcg_input = default_xcg # Initialize
try:
    xcg_str = input(f"Enter X-coordinate of moment reference point (Xcg/Xref) in meters (default: {default_xcg}): ").strip()
    xcg_input = float(xcg_str if xcg_str else str(default_xcg))
except ValueError:
    print(f"Invalid input for Xcg, using default: {default_xcg}")
    xcg_input = default_xcg
# --- END XCG INPUT ---


# --- USER INPUT FOR WAKE ITERATIONS AND SYMMETRY ---
print("\n--- Input VSPAERO Solver Settings ---")
# Wake Iterations
default_wake_iter = 10
wake_iter_input = default_wake_iter
try:
    wake_str = input(f"Enter number of Wake Iterations (default: {default_wake_iter}): ").strip()
    if wake_str: # Only parse if user entered something
        wake_iter_input = int(wake_str)
        if wake_iter_input < 0:
            print("Wake Iterations cannot be negative, using default.")
            wake_iter_input = default_wake_iter
except ValueError:
    print(f"Invalid input for Wake Iterations, using default: {default_wake_iter}")
    wake_iter_input = default_wake_iter
print(f"  Wake Iterations: {wake_iter_input}")

# Symmetry (SymmFlag)
default_symm_flag = 0 # No symmetry
symm_flag_input = default_symm_flag
symm_options = {'0': "No Symmetry", '1': "XZ Plane (Typical Aircraft)", '2': "XY Plane", '3': "YZ Plane"}
print("Symmetry Options:")
for k, v in symm_options.items(): print(f"  {k}: {v}")

try:
    symm_str = input(f"Enter Symmetry Flag (0-3, default: {default_symm_flag} - {symm_options[str(default_symm_flag)]}): ").strip()
    if symm_str:
        if symm_str in symm_options:
            symm_flag_input = int(symm_str)
        else:
            print(f"Invalid Symmetry Flag '{symm_str}'. Using default.")
            symm_flag_input = default_symm_flag
except ValueError:
    print(f"Invalid input for Symmetry Flag, using default: {default_symm_flag}")
    symm_flag_input = default_symm_flag
print(f"  Symmetry Flag: {symm_flag_input} ({symm_options[str(symm_flag_input)]})")
# --- END WAKE ITERATIONS AND SYMMETRY ---


# --- SET VSPAERO ANALYSIS PARAMETERS ---
print("\n--- Setting VSPAERO Analysis Parameters ---")

vsp.SetDoubleAnalysisInput(analysis_name, "Xcg", (xcg_input,), 0)
print(f"  VSPAERO Xcg set to: {xcg_input:.3f}m")

if sweep_mode == 'a': # Alpha Sweep
    vsp.SetIntAnalysisInput(analysis_name, "AlphaNpts", (num_alpha_points_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaStart", (alpha_start_deg_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaEnd", (alpha_end_deg_input,), 0)
    print(f"  VSPAERO Alpha set for SWEEP: {num_alpha_points_input} pt(s) from {alpha_start_deg_input:.2f} to {alpha_end_deg_input:.2f} deg.")

    if mach_input > 0:
        vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", (mach_input,), 0)
        vsp.SetDoubleAnalysisInput(analysis_name, "MachEnd", (mach_input,), 0) # End = Start for single point
        vsp.SetIntAnalysisInput(analysis_name, "MachNpts", (1,), 0)
        print(f"  VSPAERO Mach set to SINGLE value: {mach_input:.4f}")
    else: print("Warning: Mach invalid for Alpha sweep. VSPAERO uses default Mach setting.")

elif sweep_mode == 'v': # Velocity/Mach Sweep
    vsp.SetIntAnalysisInput(analysis_name, "AlphaNpts", (1,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaStart", (single_alpha_deg_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaEnd", (single_alpha_deg_input,), 0) # End = Start
    print(f"  VSPAERO Alpha set to SINGLE value: {single_alpha_deg_input:.2f} deg.")

    if mach_start_input > 0: # mach_end_input will also be >0 if num_vel_points > 1
        vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", (mach_start_input,), 0)
        vsp.SetDoubleAnalysisInput(analysis_name, "MachEnd", (mach_end_input,), 0)
        vsp.SetIntAnalysisInput(analysis_name, "MachNpts", (num_vel_points_input,), 0)
        print(f"  VSPAERO Mach set for SWEEP: {num_vel_points_input} pt(s) from {mach_start_input:.4f} to {mach_end_input:.4f}")
    else: print("Warning: Mach range invalid for Velocity sweep. VSPAERO uses default Mach settings.")

# Reynolds parameters (common to both sweep modes, using Start, End, Npts)
if re_cref_start_input > 0:
    vsp.SetDoubleAnalysisInput(analysis_name, "ReCrefStart", (re_cref_start_input,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "ReCrefEnd", (re_cref_end_input,), 0)
    vsp.SetIntAnalysisInput(analysis_name, "ReCrefNpts", (re_cref_npts_input,), 0)
    if re_cref_npts_input > 1:
        print(f"  VSPAERO ReCref set for SWEEP: {re_cref_npts_input} pt(s) from {re_cref_start_input:.0f} to {re_cref_end_input:.0f}")
    else:
        print(f"  VSPAERO ReCref set to SINGLE value: {re_cref_start_input:.0f}")
else: print("Warning: Reynolds invalid/not set. VSPAERO uses default ReCref setting(s).")


if s_calculated > 0: vsp.SetDoubleAnalysisInput(analysis_name, "Sref", (s_calculated,), 0); print(f"  Sref: {s_calculated:.6f} m^2")
else: print("Warning: Sref invalid. VSPAERO uses default Sref.")
if b_input > 0: vsp.SetDoubleAnalysisInput(analysis_name, "bref", (b_input,), 0); print(f"  bref: {b_input:.4f} m")
else: print("Warning: bref invalid. VSPAERO uses default bref.")
if c_input > 0: vsp.SetDoubleAnalysisInput(analysis_name, "cref", (c_input,), 0); print(f"  cref: {c_input:.4f} m")
else: print("Warning: cref invalid. VSPAERO uses default cref.")

# Set Wake Iterations and Symmetry
vsp.SetIntAnalysisInput(analysis_name, "WakeIter", (wake_iter_input,),0)
print(f"  WakeIter set to: {wake_iter_input}")
vsp.SetIntAnalysisInput(analysis_name, "SymmFlag", (symm_flag_input,),0)
print(f"  SymmFlag set to: {symm_flag_input} ({symm_options[str(symm_flag_input)]})")


default_ncpu = os.cpu_count() if os.cpu_count() else 4 # Suggest all cores if available
ncpu_str = input(f"NCPU (default: {default_ncpu}): ").strip(); ncpu = int(ncpu_str if ncpu_str else str(default_ncpu))
vsp.SetIntAnalysisInput(analysis_name, "NCPU", (ncpu,),0)
print(f"  NCPU set to: {ncpu}")
# --- END SET VSPAERO PARAMETERS ---






# List inputs
print("\nFinal Analysis Inputs (check carefully, especially for Propeller/Rotor mode):")
vsp.PrintAnalysisInputs(analysis_name)
print("")


# Execute
print("\tExecuting...")
rid = vsp.ExecAnalysis(analysis_name)
print("VSPAERO Execution COMPLETE")

# --- Get and Display Results from Output File ---
vsp3_file_basename = os.path.splitext(os.path.basename(vsp3_file_path_input))[0]
output_dir = os.path.dirname(vsp3_file_path_input) # Outputs go where .vsp3 file is

# Define potential polar file paths
polar_file_name_vspgeom = f"{vsp3_file_basename}_VSPGeom.polar"
polar_file_path_vspgeom = os.path.join(output_dir, polar_file_name_vspgeom)
polar_file_name_degengeom = f"{vsp3_file_basename}_DegenGeom.polar"
polar_file_path_degengeom = os.path.join(output_dir, polar_file_name_degengeom)

# Define potential rotor results file path
rotor_results_file_name = f"{vsp3_file_basename}_VSPGeom.rotor" # VSPAERO usually names it based on _VSPGeom
rotor_results_file_path = os.path.join(output_dir, rotor_results_file_name)

polar_df_to_plot = None
polar_file_processed = "" # To store the name of the file that was actually read

print(f"\n--- Attempting to read output files from: {output_dir} ---")

file_to_try_reading = None
if os.path.exists(polar_file_path_vspgeom):
    file_to_try_reading = polar_file_path_vspgeom
    polar_file_processed = polar_file_name_vspgeom
elif os.path.exists(polar_file_path_degengeom):
    print(f"Note: VSPGeom polar file not found. Attempting to use DegenGeom polar file: {polar_file_name_degengeom}")
    file_to_try_reading = polar_file_path_degengeom
    polar_file_processed = polar_file_name_degengeom
else:
    print(f"Error: No .polar file (neither VSPGeom nor DegenGeom) found for {vsp3_file_basename} in {output_dir}")
    print(f"  Checked for: '{polar_file_path_vspgeom}'")
    print(f"  And for:     '{polar_file_path_degengeom}'")


if file_to_try_reading:
    polar_df_to_plot = read_polar_file(file_to_try_reading)

    if polar_df_to_plot is not None and not polar_df_to_plot.empty:
        print(f"\n--- Plotting Data from '{polar_file_processed}' ---")
        available_cols = polar_df_to_plot.columns.tolist()

        if not available_cols:
            print("No columns found in the polar data. Cannot plot.")
        else:
            print("Available columns for plotting:")
            for i, col in enumerate(available_cols):
                print(f"  {i+1}. {col}")

            x_axis_col_input = ""
            x_axis_valid = False
            
            # Adjust default X-axis based on sweep_mode
            default_x_col = None
            if sweep_mode == 'a':
                default_x_col = 'Alpha' if 'Alpha' in available_cols else ('AoA' if 'AoA' in available_cols else None)
            elif sweep_mode == 'v':
                default_x_col = 'Mach' if 'Mach' in available_cols else None # VSPAERO .polar has Mach, not usually Vinf
            if not default_x_col and available_cols: # Fallback
                 default_x_col = available_cols[0]


            while not x_axis_valid:
                prompt_x = f"Enter the NAME or NUMBER for the X-AXIS (default: {default_x_col if default_x_col else 'None'}): "
                x_choice_str = input(prompt_x).strip()
                if not x_choice_str and default_x_col:
                    x_axis_col_input = default_x_col; x_axis_valid = True; print(f"Using default X-axis: {x_axis_col_input}")
                elif x_choice_str.isdigit():
                    try: x_idx = int(x_choice_str) - 1
                    except ValueError: print("Invalid number."); continue
                    if 0 <= x_idx < len(available_cols): x_axis_col_input = available_cols[x_idx]; x_axis_valid = True
                    else: print("Invalid number choice.")
                elif x_choice_str in available_cols: x_axis_col_input = x_choice_str; x_axis_valid = True
                else: print(f"Column '{x_choice_str}' not found.")

            y_axis_col_input = ""
            y_axis_valid = False
            if x_axis_valid:
                default_y_candidates = [col for col in available_cols if col != x_axis_col_input]
                default_y_col_cl_candidate = next((c for c in default_y_candidates if c.upper() == 'CL'), None)
                default_y_col = default_y_col_cl_candidate if default_y_col_cl_candidate else \
                                (default_y_candidates[0] if default_y_candidates else (available_cols[0] if available_cols else None))

                while not y_axis_valid:
                    prompt_y = f"Enter NAME or NUMBER for Y-AXIS (vs '{x_axis_col_input}', default: {default_y_col if default_y_col else 'None'}): "
                    y_choice_str = input(prompt_y).strip()
                    if not y_choice_str and default_y_col:
                        y_axis_col_input = default_y_col; y_axis_valid = True; print(f"Using default Y-axis: {y_axis_col_input}")
                    elif y_choice_str.isdigit():
                        try: y_idx = int(y_choice_str) - 1
                        except ValueError: print("Invalid number."); continue
                        if 0 <= y_idx < len(available_cols): y_axis_col_input = available_cols[y_idx]; y_axis_valid = True
                        else: print("Invalid number choice.")
                    elif y_choice_str in available_cols: y_axis_col_input = y_choice_str; y_axis_valid = True
                    else: print(f"Column '{y_choice_str}' not found.")
                    if y_axis_valid and y_axis_col_input == x_axis_col_input: print(f"Note: Plotting '{y_axis_col_input}' against itself.")

                if y_axis_valid:
                    plot_polar_data(polar_df_to_plot, x_axis_col_input, y_axis_col_input, polar_file_processed)
                else:
                    print("Skipping polar data plotting as Y-axis was not specified or valid.")
            else:
                print("Skipping polar data plotting as X-axis was not specified or valid.")
    elif polar_df_to_plot is not None and polar_df_to_plot.empty:
        print(f"The polar file '{file_to_try_reading}' was read, but it appears to be empty or could not be parsed correctly by 'read_polar_file'.")

if is_prop_analysis:
    if os.path.exists(rotor_results_file_path):
        print(f"\nFound rotor results file: {rotor_results_file_path}")
        print("Rotor performance data (Thrust, Torque, Power, etc.) would be in this file.")
        try:
            with open(rotor_results_file_path, 'r') as rf:
                print("\nFirst few lines of .rotor file:")
                for i in range(10): # Print more lines for better context
                    line = rf.readline().strip()
                    if not line: break
                    print(line)
                print("...")
        except Exception as e:
            print(f"Could not read .rotor file: {e}")
    else:
        print(f"\nNo .rotor file ({rotor_results_file_name}) found. Propeller analysis might have failed, not run, or produced different output names.")
        # prop_geom_name_input might not be defined if is_prop_analysis is false, handle this.
        prop_name_for_msg = prop_geom_name_input if 'prop_geom_name_input' in locals() and prop_geom_name_input else 'N/A'
        print(f"  Ensure the propeller component name ('{prop_name_for_msg}') was correctly set and is active in VSPAERO.")


print("\nAnalysis script finished.")