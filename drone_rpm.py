import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.style.use("paper.mplstyle")

pt = 1./72.27

jour_sizes = {"PRD": {"onecol": 483.69*pt},
             }

my_width = jour_sizes["PRD"]["onecol"]
golden = (1 + 5 ** 0.5) / 2

# --- Constants and Configuration ---
RHO_AIR = 1.225  # kg/m^3 (Density of air)
MU_AIR = 1.81e-5 # Pa.s (Dynamic viscosity of air at ~15°C)

# Duct parameters (based on the "thin annulus" assumption)
D_MEAN_DUCT_M = 0.2  # Mean diameter of the annular wing/duct in meters
R_MEAN_DUCT_M = D_MEAN_DUCT_M / 2.0 # Mean radius of the duct
L_DUCT_M = 0.1 # Chord length of the NACA0012 airfoil, treated as the axial length for friction

RE_PHI_CRITICAL = 60.0 # Rotational Reynolds number for C_mc transition

# Propeller Regression Coefficients
A_T_REG = 6.2461e-11
B_T_REG = 2.1652e-07
C_T_REG = -8.8256e-04

A_THRUST_RPM_REG = 5.7347e-09
B_THRUST_RPM_REG = 1.5587e-05
C_THRUST_RPM_REG = -1.8603e-01

A_POWER_THRUST_REG = 4.8792e+00
B_POWER_THRUST_REG = 12.7298
C_POWER_THRUST_REG = -0.0332

TARGET_THRUST_N = 0.2

# --- USER ADJUSTABLE PLOT SETTING ---
# Set Y_AXIS_LOWER_LIMIT_PLOT to a numeric value (e.g., 0.1, 0.5) to set the y-axis start.
# Set to None for automatic y-axis lower limit.
Y_AXIS_LOWER_LIMIT_PLOT = 0.1 # Example: start y-axis at 0.1 N

# Global list to store thrust values during brentq iterations for plotting
thrust_convergence_history = []

# --- Helper Functions ---
# ... (all your helper functions remain unchanged) ...
def solve_quadratic_positive_root(a, b, c):
    if a == 0:
        return -c / b if b != 0 and -c / b > 0 else None # Check for positive root
    delta = b**2 - 4*a*c
    if delta < 0: return None
    sqrt_delta = np.sqrt(delta)
    roots = [(-b + sqrt_delta) / (2*a), (-b - sqrt_delta) / (2*a)]
    positive_roots = [r for r in roots if r > 1e-9] # Filter for positive roots (small epsilon for float comparison)
    return max(positive_roots) if positive_roots else None

# def get_torque_for_thrust_regression(target_thrust_n):
#     return solve_quadratic_positive_root(A_TH_REG, B_TH_REG, C_TH_REG - target_thrust_n)

def get_n_prop_ground_for_thrust_rpm_regression(target_thrust_n):
    """Calculates propeller RPM required to achieve target_thrust_n using Thrust vs RPM regression."""
    # Solves: target_thrust_n = A_THRUST_RPM_REG * RPM^2 + B_THRUST_RPM_REG * RPM + C_THRUST_RPM_REG
    # which is: A_THRUST_RPM_REG * RPM^2 + B_THRUST_RPM_REG * RPM + (C_THRUST_RPM_REG - target_thrust_n) = 0
    return solve_quadratic_positive_root(A_THRUST_RPM_REG, B_THRUST_RPM_REG, C_THRUST_RPM_REG - target_thrust_n)

def calculate_prop_torque_from_n_prop_ground(n_prop_ground_rpm):
    rpm_for_calc = max(0, n_prop_ground_rpm) # Handles negative RPM by treating as 0
    torque_nm = (A_T_REG * rpm_for_calc**2) + (B_T_REG * rpm_for_calc) + C_T_REG
    return max(0, torque_nm) # Torque cannot be negative

def calculate_thrust_from_n_prop_ground(n_prop_ground_rpm):
    """Calculates thrust from propeller RPM using Thrust vs RPM regression."""
    rpm_for_calc = max(0, n_prop_ground_rpm) # Handles negative RPM by treating as 0
    # Thrust = A_THRUST_RPM_REG * RPM^2 + B_THRUST_RPM_REG * RPM + C_THRUST_RPM_REG
    thrust_n = (A_THRUST_RPM_REG * rpm_for_calc**2) + \
               (B_THRUST_RPM_REG * rpm_for_calc) + \
               C_THRUST_RPM_REG
    return max(0, thrust_n) # Thrust cannot be negative

def calculate_power_from_thrust(thrust_n):
    """Calculates estimated power from thrust using Power vs Thrust regression."""
    # Power = A_POWER_THRUST_REG * Thrust^2 + B_POWER_THRUST_REG * Thrust + C_POWER_THRUST_REG
    power_w = (A_POWER_THRUST_REG * thrust_n**2) + \
              (B_POWER_THRUST_REG * thrust_n) + \
              C_POWER_THRUST_REG
    return max(0, power_w) # Power cannot be negative


def solve_turbulent_cmc(Re_phi):
    if Re_phi < 1e-9: return 0.0
    # Equation for 1/sqrt(C_mc) where x = sqrt(C_mc)
    # 1/x = -0.8572 + 1.250 * ln(Re_phi * x)
    def eq_arr(x_arr): # x = sqrt(C_mc)
        x = max(1e-9, x_arr[0])
        val_log = Re_phi * x
        if val_log <= 1e-9: return 1e12
        try: log_term = np.log(val_log) # Natural log
        except: return 1e12
        return 1.0/x - (-0.8572 + 1.250*log_term)

    def eq_scalar(x): # x = sqrt(C_mc)
        x = max(1e-9, x)
        val_log = Re_phi * x
        if val_log <= 1e-9: return 1e12
        try: log_term = np.log(val_log)
        except: return 1e12
        return 1.0/x - (-0.8572 + 1.250*log_term)

    try:
        fa, fb = eq_scalar(0.01), eq_scalar(0.5) # Bounds for sqrt(C_mc)
        if np.isfinite(fa) and np.isfinite(fb) and fa*fb < 0:
            return brentq(eq_scalar, 0.01, 0.5, xtol=1e-7, rtol=1e-7)**2 # Result is sqrt(C_mc), so square it

        sol, _, ier, _ = fsolve(eq_arr, [0.07], xtol=1e-7, full_output=True) # Initial guess for sqrt(C_mc)
        return sol[0]**2 if ier == 1 else 0.005 # Fallback C_mc
    except:
        return 0.005 # Fallback C_mc on any error

def calculate_duct_drag_torque_and_cmc(Omega_duct_rad_s):
    if Omega_duct_rad_s < 1e-9 : return 0.0, 0.0, 0.0

    Re_phi_duct = (RHO_AIR * Omega_duct_rad_s * R_MEAN_DUCT_M**2) / MU_AIR

    if Re_phi_duct < 1e-9 : return 0.0, 0.0, Re_phi_duct

    C_mc_duct_single_surface = 0.0
    if Re_phi_duct < RE_PHI_CRITICAL: # Laminar
        if Re_phi_duct == 0: return 0.0, 0.0, Re_phi_duct
        C_mc_duct_single_surface = 8.0 / Re_phi_duct
    else: # Turbulent
        C_mc_duct_single_surface = solve_turbulent_cmc(Re_phi_duct)

    if C_mc_duct_single_surface < 0: C_mc_duct_single_surface = 1e-6

    total_duct_torque = C_mc_duct_single_surface * np.pi * RHO_AIR * (Omega_duct_rad_s**2) * (R_MEAN_DUCT_M**4) * L_DUCT_M

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
            nd_low_inner = 0.0
            nd_high_inner = n_motor_setting_trial_rpm * 1.1
            if nd_high_inner < 100 and n_motor_setting_trial_rpm > 0: nd_high_inner = max(100, n_motor_setting_trial_rpm + 10)
            elif nd_high_inner < 1: nd_high_inner = 100 # Case if n_motor_setting_trial_rpm is 0 or very small


            val_low_inner = inner_torque_balance_objective(nd_low_inner)
            val_high_inner = inner_torque_balance_objective(nd_high_inner)

            cnt = 0
            while val_high_inner > 0 and cnt < 5:
                nd_high_inner *= 1.5
                val_high_inner = inner_torque_balance_objective(nd_high_inner)
                cnt += 1

            if val_low_inner * val_high_inner <= 0:
                if abs(val_low_inner) < 1e-7: n_duct_eq_rpm_local = nd_low_inner
                elif abs(val_high_inner) < 1e-7: n_duct_eq_rpm_local = nd_high_inner
                else:
                    n_duct_eq_rpm_local = brentq(inner_torque_balance_objective, nd_low_inner, nd_high_inner, xtol=1e-5, rtol=1e-5)
    except (ValueError, RuntimeError):
        n_duct_eq_rpm_local = np.nan

    if np.isnan(n_duct_eq_rpm_local) or n_duct_eq_rpm_local < -1e-3:
        error_val = -TARGET_THRUST_N if n_motor_setting_trial_rpm < 5000 else 1e6
        thrust_achieved_for_failed_case = error_val + TARGET_THRUST_N
        return error_val, n_duct_eq_rpm_local, np.nan, np.nan, thrust_achieved_for_failed_case

    n_duct_eq_rpm_local = max(0, n_duct_eq_rpm_local)

    n_prop_ground_eq_rpm_local = n_motor_setting_trial_rpm - n_duct_eq_rpm_local

    prop_torque_eq_local = calculate_prop_torque_from_n_prop_ground(n_prop_ground_eq_rpm_local)
    # thrust_achieved_local = calculate_thrust_from_prop_torque(prop_torque_eq_local) # Old version
    thrust_achieved_local = calculate_thrust_from_n_prop_ground(n_prop_ground_eq_rpm_local) # New version

    return thrust_achieved_local - TARGET_THRUST_N, n_duct_eq_rpm_local, n_prop_ground_eq_rpm_local, prop_torque_eq_local, thrust_achieved_local

def outer_objective_for_brentq(n_motor_setting_trial_rpm_scalar):
    error, _, _, _, thrust_achieved = calculate_system_state_and_thrust_error(n_motor_setting_trial_rpm_scalar)
    thrust_convergence_history.append(thrust_achieved)
    return error

# --- Main Process: Solve for N_motor_setting_rpm to achieve TARGET_THRUST_N ---
# ... (main solving process remains unchanged) ...
thrust_convergence_history.clear()

print(f"Attempting to find motor RPM setting for Target Thrust: {TARGET_THRUST_N:.3f} N")
n_motor_low_bound_est = 2000
# Old estimation based on torque regression
# torque_for_target_thrust_est = get_torque_for_thrust_regression(TARGET_THRUST_N)
# if torque_for_target_thrust_est is not None and torque_for_target_thrust_est > 1e-6:
#     n_pg_for_torque_est = get_n_prop_ground_for_torque_regression(torque_for_target_thrust_est)
#     if n_pg_for_torque_est is not None and n_pg_for_torque_est > 0:
#         n_motor_low_bound_est = max(n_motor_low_bound_est, n_pg_for_torque_est)

# New estimation based on Thrust vs RPM regression
n_pg_for_thrust_est = get_n_prop_ground_for_thrust_rpm_regression(TARGET_THRUST_N)
if n_pg_for_thrust_est is not None and n_pg_for_thrust_est > 0:
    # This estimates the propeller RPM. The motor RPM will be this + duct RPM.
    # For a low bound, we can assume duct RPM is small, or just use n_pg as a starting point.
    # If the duct requires significant RPM, this estimate might be too low,
    # but brentq should still work if the range [low, high] brackets the root.
    n_motor_low_bound_est = max(n_motor_low_bound_est, n_pg_for_thrust_est)


n_motor_high_bound_est = 25000 # User updated this from 60000

n_motor_solution_rpm = np.nan
final_n_pg_rpm, final_n_dg_rpm = np.nan, np.nan
final_prop_torque, final_thrust = np.nan, np.nan
final_duct_cmc, final_duct_torque = np.nan, np.nan
brentq_results_obj = None

try:
    err_at_low_bound = outer_objective_for_brentq(n_motor_low_bound_est)
    err_at_high_bound = outer_objective_for_brentq(n_motor_high_bound_est)

    if err_at_low_bound * err_at_high_bound <= 0:
        n_motor_solution_rpm, brentq_results_obj = brentq(
            outer_objective_for_brentq,
            n_motor_low_bound_est,
            n_motor_high_bound_est,
            xtol=0.1,
            rtol=1e-3,
            full_output=True
        )

        _, final_n_dg_rpm, final_n_pg_rpm, final_prop_torque, final_thrust = \
            calculate_system_state_and_thrust_error(n_motor_solution_rpm)

        final_omega_duct_rads = final_n_dg_rpm * (2*np.pi) / 60.0
        final_duct_torque_check, final_duct_cmc, _ = calculate_duct_drag_torque_and_cmc(final_omega_duct_rads)
        final_duct_torque = final_duct_torque_check
    else:
        print(f"Could not bracket solution for N_motor_setting. Err_low={err_at_low_bound:.2e} (at {n_motor_low_bound_est:.0f} RPM), Err_high={err_at_high_bound:.2e} (at {n_motor_high_bound_est:.0f} RPM)")

except (ValueError, RuntimeError) as e_outer:
    print(f"Outer solver (brentq) failed: {e_outer}")
except Exception as e_outer_unexpected:
    print(f"Unexpected error in main solving process: {e_outer_unexpected}")

# --- Output Final Solution ---
# ... (output remains unchanged) ...
if not np.isnan(n_motor_solution_rpm) and brentq_results_obj and brentq_results_obj.converged:
    print("\n--- Solution Found for Target Thrust ---")
    print(f"Brentq iterations: {brentq_results_obj.iterations}, Function calls (total): {brentq_results_obj.function_calls + 2}")
    print(f"Target Propeller Thrust: {TARGET_THRUST_N:.4f} N")
    print(f"Required Motor Setting RPM (N_motor_setting): {n_motor_solution_rpm:.2f} RPM")
    print(f"Resulting Propeller RPM (ground, N_pg): {final_n_pg_rpm:.2f} RPM")
    print(f"Resulting Duct RPM (ground, N_dg): {final_n_dg_rpm:.2f} RPM")
    print(f"Achieved Propeller Torque (@N_pg): {final_prop_torque:.4f} Nm")
    print(f"Achieved Propeller Thrust (@N_pg): {final_thrust:.4f} N (Error: {(final_thrust - TARGET_THRUST_N):.2e} N)")
    
    # Calculate and print estimated power
    estimated_power_w = calculate_power_from_thrust(final_thrust)
    print(f"Estimated Power Draw (@Achieved Thrust): {estimated_power_w:.2f} W")
    
    print(f"Duct C_mc (single surface ref @N_dg): {final_duct_cmc:.4f}")
    print(f"Total Duct Torque (both surfaces @N_dg): {final_duct_torque:.4f} Nm")
    if not (np.isnan(final_prop_torque) or np.isnan(final_duct_torque)):
      print(f"Torque Balance Check: Prop Torque - Duct Torque = {final_prop_torque - final_duct_torque:.2e} Nm")
else:
    print("\n--- No Solution Found for Target Thrust ---")
    if brentq_results_obj:
        print(f"Brentq convergence status: {brentq_results_obj.converged}")
        print(f"Brentq iterations: {brentq_results_obj.iterations}, Function calls (total): {brentq_results_obj.function_calls + 2}")

# --- Plotting Convergence ---
if thrust_convergence_history:
    iterations = range(1, len(thrust_convergence_history) + 1)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(iterations, thrust_convergence_history, marker='o', linestyle='-', label='Thrust per Iteration')
    ax.axhline(y=TARGET_THRUST_N, color='r', linestyle='--', label=f'Target Thrust ({TARGET_THRUST_N:.2f} N)')

    if not np.isnan(final_thrust):
        ax.axhline(y=final_thrust, color='g', linestyle=':', linewidth=2, label=f'Final Achieved Thrust ({final_thrust:.4f} N)')

    ax.set_title('Thrust Convergence during Outer Solver Iterations')
    ax.set_xlabel('Outer Solver Iteration')
    ax.set_ylabel('Calculated Thrust (N)')

    current_linthresh = max(0.01, TARGET_THRUST_N / 10.0)
    ax.set_yscale('symlog', linthresh=current_linthresh)

    current_ylim = ax.get_ylim()
    bottom_ylim_to_set = current_ylim[0]

    if Y_AXIS_LOWER_LIMIT_PLOT is not None:
        bottom_ylim_to_set = Y_AXIS_LOWER_LIMIT_PLOT
    
    if bottom_ylim_to_set < current_ylim[1]:
         ax.set_ylim(bottom=bottom_ylim_to_set, top=current_ylim[1])
    else:
         ax.set_ylim(bottom=bottom_ylim_to_set)

    # Enable minor ticks ONLY on the Y-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # If you wanted to explicitly turn OFF minor ticks on X-axis (though they might be off by default if not triggered):
    # from matplotlib.ticker import NullLocator
    # ax.xaxis.set_minor_locator(NullLocator())


    # Grid settings:
    # Major grid lines on both X and Y axes
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth='0.5', color='gray')
    # Minor grid lines ONLY on Y-axis
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth='0.5', color='lightgray')
    fig = plt.figure(figsize = (my_width, my_width/golden))
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No thrust convergence data to plot (thrust_convergence_history is empty).")