import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
import re

# --- Constants and Configuration ---
RHO_AIR = 1.225  # kg/m^3 (Density of air)
MU_AIR = 1.81e-5 # Pa.s (Dynamic viscosity of air at ~15°C)

# Duct parameters (based on the "thin annulus" assumption)
D_MEAN_DUCT_M = 0.255  # Mean diameter of the annular wing/duct in meters
R_MEAN_DUCT_M = D_MEAN_DUCT_M / 2.0 # Mean radius of the duct
L_DUCT_M = 0.128 # Chord length of the NACA0012 airfoil, treated as the axial length for friction

RE_PHI_CRITICAL = 60.0 # Rotational Reynolds number for C_mc transition

# Propeller Regression Coefficients
A_T_REG = 1.3197e-10
B_T_REG = -5.1746e-08
C_T_REG = -3.4041e-03
A_TH_REG = 3.3537e+01
B_TH_REG = 77.1731
C_TH_REG = -0.3048

TARGET_THRUST_N = 0.9

# --- Helper Functions ---

def solve_quadratic_positive_root(a, b, c):
    if a == 0:
        return -c / b if b != 0 and -c / b > 0 else None
    delta = b**2 - 4*a*c
    if delta < 0: return None
    sqrt_delta = np.sqrt(delta)
    roots = [(-b + sqrt_delta) / (2*a), (-b - sqrt_delta) / (2*a)]
    positive_roots = [r for r in roots if r > 1e-9]
    return max(positive_roots) if positive_roots else None

def get_torque_for_thrust_regression(target_thrust_n):
    return solve_quadratic_positive_root(A_TH_REG, B_TH_REG, C_TH_REG - target_thrust_n)

def get_n_prop_ground_for_torque_regression(target_torque_nm):
    return solve_quadratic_positive_root(A_T_REG, B_T_REG, C_T_REG - target_torque_nm)

def calculate_prop_torque_from_n_prop_ground(n_prop_ground_rpm):
    rpm_for_calc = max(0, n_prop_ground_rpm)
    torque_nm = (A_T_REG * rpm_for_calc**2) + (B_T_REG * rpm_for_calc) + C_T_REG
    return max(0, torque_nm)

def calculate_thrust_from_prop_torque(torque_nm):
    if torque_nm <= 1e-9: return 0.0
    thrust_n = (A_TH_REG * torque_nm**2) + (B_TH_REG * torque_nm) + C_TH_REG
    return max(0, thrust_n)

def solve_turbulent_cmc(Re_phi):
    if Re_phi < 1e-9: return 0.0
    def eq_arr(x_arr):
        x = max(1e-9, x_arr[0])
        val_log = Re_phi * x
        if val_log <= 1e-9: return 1e12
        try: log_term = np.log(val_log)
        except: return 1e12
        return 1.0/x - (-0.8572 + 1.250*log_term)
    def eq_scalar(x):
        x = max(1e-9, x)
        val_log = Re_phi * x
        if val_log <= 1e-9: return 1e12
        try: log_term = np.log(val_log)
        except: return 1e12
        return 1.0/x - (-0.8572 + 1.250*np.log(val_log))
    try:
        fa, fb = eq_scalar(0.01), eq_scalar(0.5)
        if np.isfinite(fa) and np.isfinite(fb) and fa*fb < 0:
            return brentq(eq_scalar, 0.01, 0.5, xtol=1e-7, rtol=1e-7)**2
        sol, _, ier, _ = fsolve(eq_arr, [0.07], xtol=1e-7, full_output=True)
        return sol[0]**2 if ier == 1 else 0.005 # Fallback
    except: return 0.005 # Fallback

def calculate_duct_drag_torque_and_cmc(Omega_duct_rad_s):
    """
    Calculates the total skin friction drag torque on the thin annular duct,
    its effective moment coefficient (C_mc_duct), and its rotational Reynolds number (Re_phi_duct).
    Omega_duct_rad_s is the duct's rotational speed in rad/s relative to the ground.
    The C_mc_duct is calculated for a cylinder of R_MEAN_DUCT_M.
    The total_duct_torque accounts for both "sides" of the thin annulus.
    """
    if Omega_duct_rad_s < 1e-9 : return 0.0, 0.0, 0.0

    Re_phi_duct = (RHO_AIR * Omega_duct_rad_s * R_MEAN_DUCT_M**2) / MU_AIR

    if Re_phi_duct < 1e-9 : return 0.0, 0.0, Re_phi_duct

    C_mc_duct_single_surface = 0.0 # This C_mc is for one reference surface area
    if Re_phi_duct < RE_PHI_CRITICAL:
        if Re_phi_duct == 0: return 0.0, 0.0, Re_phi_duct
        C_mc_duct_single_surface = 8.0 / Re_phi_duct
    else:
        C_mc_duct_single_surface = solve_turbulent_cmc(Re_phi_duct)

    if C_mc_duct_single_surface < 0: C_mc_duct_single_surface = 1e-6

    # Torque for ONE cylindrical surface:
    # Torque_one_surface = C_mc_single_surface * 0.5 * np.pi * rho * Omega^2 * R^4 * L
    # For a thin annulus, we have two such effective surfaces (inner and outer are very close to mean).
    # So, Total_Duct_Torque = 2 * Torque_one_surface_at_mean_radius
    # Total_Duct_Torque = 2 * (C_mc_duct_single_surface * 0.5 * np.pi * RHO_AIR * (Omega_duct_rad_s**2) * (R_MEAN_DUCT_M**4) * L_DUCT_M)
    total_duct_torque = C_mc_duct_single_surface * np.pi * RHO_AIR * (Omega_duct_rad_s**2) * (R_MEAN_DUCT_M**4) * L_DUCT_M
    
    # The C_mc_duct reported will be the C_mc_duct_single_surface, as that's the standard coefficient.
    # The torque calculation then correctly uses it.
    return total_duct_torque, C_mc_duct_single_surface, Re_phi_duct

# --- Function for Outer Solver (Thrust Target) ---

def calculate_system_state_and_thrust_error(n_motor_setting_trial_rpm):
    def inner_torque_balance_objective(n_duct_ground_trial_rpm):
        if n_duct_ground_trial_rpm < 0: return 1e12
        n_pg_trial = n_motor_setting_trial_rpm - n_duct_ground_trial_rpm
        prop_torque_trial = calculate_prop_torque_from_n_prop_ground(n_pg_trial)
        omega_duct_trial_rads = n_duct_ground_trial_rpm * (2*np.pi) / 60.0
        duct_torque_trial, _, _ = calculate_duct_drag_torque_and_cmc(omega_duct_trial_rads)
        return prop_torque_trial - duct_torque_trial

    n_duct_eq_rpm_local = np.nan
    try:
        if calculate_prop_torque_from_n_prop_ground(n_motor_setting_trial_rpm) < 1e-7 :
             n_duct_eq_rpm_local = 0.0
        else:
            nd_low_inner, nd_high_inner = 0.0, n_motor_setting_trial_rpm * 2.0
            val_low_inner, val_high_inner = inner_torque_balance_objective(nd_low_inner), inner_torque_balance_objective(nd_high_inner)
            cnt = 0
            while val_high_inner > 0 and cnt < 5:
                nd_high_inner *= 1.5; val_high_inner = inner_torque_balance_objective(nd_high_inner); cnt += 1
            if val_low_inner * val_high_inner <= 0:
                if abs(val_low_inner) < 1e-7: n_duct_eq_rpm_local = nd_low_inner
                elif abs(val_high_inner) < 1e-7: n_duct_eq_rpm_local = nd_high_inner
                else: n_duct_eq_rpm_local = brentq(inner_torque_balance_objective, nd_low_inner, nd_high_inner, xtol=1e-5, rtol=1e-5)
    except (ValueError, RuntimeError): n_duct_eq_rpm_local = np.nan

    if np.isnan(n_duct_eq_rpm_local) or n_duct_eq_rpm_local < 0:
        return -TARGET_THRUST_N if n_motor_setting_trial_rpm < 5000 else 1e6 

    n_duct_eq_rpm_local = max(0, n_duct_eq_rpm_local)
    n_prop_ground_eq_rpm_local = n_motor_setting_trial_rpm - n_duct_eq_rpm_local
    prop_torque_eq_local = calculate_prop_torque_from_n_prop_ground(n_prop_ground_eq_rpm_local)
    thrust_achieved_local = calculate_thrust_from_prop_torque(prop_torque_eq_local)
    return thrust_achieved_local - TARGET_THRUST_N, n_duct_eq_rpm_local, n_prop_ground_eq_rpm_local, prop_torque_eq_local, thrust_achieved_local

def outer_objective_for_brentq(n_motor_setting_trial_rpm_scalar):
    error, _, _, _, _ = calculate_system_state_and_thrust_error(n_motor_setting_trial_rpm_scalar)
    return error

# --- Main Process: Solve for N_motor_setting_rpm to achieve TARGET_THRUST_N ---
print(f"Attempting to find motor RPM setting for Target Thrust: {TARGET_THRUST_N:.3f} N")
n_motor_low_bound_est = 2000
torque_for_target_thrust = get_torque_for_thrust_regression(TARGET_THRUST_N)
if torque_for_target_thrust is not None and torque_for_target_thrust > 1e-6:
    n_pg_for_torque_est = get_n_prop_ground_for_torque_regression(torque_for_target_thrust)
    if n_pg_for_torque_est is not None and n_pg_for_torque_est > 0:
        n_motor_low_bound_est = max(n_motor_low_bound_est, n_pg_for_torque_est)
n_motor_high_bound_est = 60000
n_motor_solution_rpm, final_n_pg_rpm, final_n_dg_rpm = np.nan, np.nan, np.nan
final_prop_torque, final_thrust, final_duct_cmc, final_duct_torque = np.nan, np.nan, np.nan, np.nan

try:
    err_at_low_bound = outer_objective_for_brentq(n_motor_low_bound_est)
    err_at_high_bound = outer_objective_for_brentq(n_motor_high_bound_est)
    if err_at_low_bound * err_at_high_bound <= 0:
        n_motor_solution_rpm = brentq(outer_objective_for_brentq, n_motor_low_bound_est, n_motor_high_bound_est, xtol=0.1, rtol=1e-3)
        _, final_n_dg_rpm, final_n_pg_rpm, final_prop_torque, final_thrust = calculate_system_state_and_thrust_error(n_motor_solution_rpm)
        final_omega_duct_rads = final_n_dg_rpm * (2*np.pi) / 60.0
        final_duct_torque_check, final_duct_cmc, _ = calculate_duct_drag_torque_and_cmc(final_omega_duct_rads)
        final_duct_torque = final_duct_torque_check
    else:
        print(f"Could not bracket solution for N_motor_setting. Err_low={err_at_low_bound:.2e}, Err_high={err_at_high_bound:.2e}")
except (ValueError, RuntimeError) as e_outer: print(f"Outer solver (brentq) failed: {e_outer}")
except Exception as e_outer_unexpected: print(f"Unexpected error in main solving process: {e_outer_unexpected}")

# --- Output Final Solution ---
if not np.isnan(n_motor_solution_rpm):
    print("\n--- Solution Found for Target Thrust ---")
    print(f"Target Propeller Thrust: {TARGET_THRUST_N:.4f} N")
    print(f"Required Motor Setting RPM (N_prop_body): {n_motor_solution_rpm:.2f} RPM")
    print(f"Resulting Propeller RPM (ground, N_pg): {final_n_pg_rpm:.2f} RPM")
    print(f"Resulting Duct RPM (ground, N_dg): {final_n_dg_rpm:.2f} RPM")
    print(f"Achieved Propeller Torque (@N_pg): {final_prop_torque:.4f} Nm")
    print(f"Achieved Propeller Thrust (@N_pg): {final_thrust:.4f} N (Error: {(final_thrust - TARGET_THRUST_N):.2e} N)")
    print(f"Duct C_mc (single surface ref @N_dg): {final_duct_cmc:.4f}") # Clarified C_mc meaning
    print(f"Total Duct Torque (both surfaces @N_dg): {final_duct_torque:.4f} Nm")
else:
    print("\n--- No Solution Found for Target Thrust ---")