import re
import numpy as np
import matplotlib.pyplot as plt

# Conversion factors
INLBF_TO_NM = 0.112984829
LBF_TO_N = 4.4482216153

def parse_prop_data(file_path, target_v_mph=33.0):
    """
    Parses the propeller data file to extract RPM, Torque (In-Lbf), and Thrust (Lbf)
    for the velocity closest to target_v_mph for each RPM section.

    Args:
        file_path (str): Path to the data file.
        target_v_mph (float): The target velocity in MPH.

    Returns:
        list: A list of tuples, where each tuple is
              (rpm, torque_in_lbf, thrust_lbf, actual_v_mph_for_data).
    """
    collected_data = []
    current_rpm = None
    min_v_diff_for_current_rpm = float('inf')
    best_torque_for_current_rpm = None
    best_thrust_for_current_rpm = None
    actual_v_at_best_data = None

    rpm_regex = re.compile(r"^\s*PROP RPM =\s*(\d+)")

    with open(file_path, 'r') as f:
        for line in f:
            rpm_match = rpm_regex.match(line)
            if rpm_match:
                if current_rpm is not None and best_torque_for_current_rpm is not None and best_thrust_for_current_rpm is not None:
                    collected_data.append((current_rpm, best_torque_for_current_rpm, best_thrust_for_current_rpm, actual_v_at_best_data))

                current_rpm = int(rpm_match.group(1))
                min_v_diff_for_current_rpm = float('inf')
                best_torque_for_current_rpm = None
                best_thrust_for_current_rpm = None
                actual_v_at_best_data = None
                continue

            if current_rpm is None:
                continue

            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] and parts[0][0].isdigit():
                try:
                    v_mph = float(parts[0])
                    torque_in_lbf = float(parts[6]) # Torque(In-Lbf) is 7th column
                    thrust_lbf = float(parts[7])    # Thrust(Lbf) is 8th column

                    v_diff = abs(v_mph - target_v_mph)

                    if v_diff < min_v_diff_for_current_rpm:
                        min_v_diff_for_current_rpm = v_diff
                        best_torque_for_current_rpm = torque_in_lbf
                        best_thrust_for_current_rpm = thrust_lbf
                        actual_v_at_best_data = v_mph
                except ValueError:
                    pass
                except IndexError:
                    pass

        if current_rpm is not None and best_torque_for_current_rpm is not None and best_thrust_for_current_rpm is not None:
            collected_data.append((current_rpm, best_torque_for_current_rpm, best_thrust_for_current_rpm, actual_v_at_best_data))

    return collected_data

def main():
    file_path = r'C:\Users\kilianseibl\Downloads\PERFILES_WEB-202503\PERFILES_WEB\PERFILES2\PER3_4x4E-3.txt'
    target_velocity_mph = 33.0

    extracted_points = parse_prop_data(file_path, target_velocity_mph)

    if not extracted_points:
        print(f"No data points found near {target_velocity_mph} mph.")
        return

    rpms = []
    torques_nm = []
    thrusts_n = []

    print(f"Extracted Data closest to V = {target_velocity_mph} mph:")
    print("--------------------------------------------------------------------------------------")
    print("RPM    | Actual V (mph) | Torque (In-Lbf) | Torque (N-m) | Thrust (Lbf) | Thrust (N)")
    print("-------|----------------|-----------------|--------------|--------------|-----------")
    for rpm, torque_in_lbf, thrust_lbf, actual_v in extracted_points:
        torque_newton_meter = torque_in_lbf * INLBF_TO_NM
        thrust_newton = thrust_lbf * LBF_TO_N

        rpms.append(rpm)
        torques_nm.append(torque_newton_meter)
        thrusts_n.append(thrust_newton)
        print(f"{rpm:<7}| {actual_v:<14.2f} | {torque_in_lbf:<15.3f} | {torque_newton_meter:<12.4f} | {thrust_lbf:<12.3f} | {thrust_newton:.3f}")
    print("--------------------------------------------------------------------------------------")

    rpm_np = np.array(rpms)
    torque_nm_np = np.array(torques_nm)
    thrust_n_np = np.array(thrusts_n)

    # --- Plot 1: Torque vs RPM ---
    if len(rpms) >= 3: # Need at least 3 points for quadratic
        degree_torque_rpm = 2
        coeffs_torque_rpm = np.polyfit(rpm_np, torque_nm_np, degree_torque_rpm)
        poly_func_torque_rpm = np.poly1d(coeffs_torque_rpm)
        print(f"\nRegression for Torque vs RPM (Torque = a*RPM^2 + b*RPM + c):")
        print(f"a = {coeffs_torque_rpm[0]:.4e}")
        print(f"b = {coeffs_torque_rpm[1]:.4e}")
        print(f"c = {coeffs_torque_rpm[2]:.4e}")
        print(f"Equation: Torque_Nm ≈ ({coeffs_torque_rpm[0]:.3e})*RPM^2 + ({coeffs_torque_rpm[1]:.3e})*RPM + ({coeffs_torque_rpm[2]:.3e})")

        rpm_fit_line = np.linspace(min(rpm_np), max(rpm_np), 200)
        torque_fit_line_rpm = poly_func_torque_rpm(rpm_fit_line)

        plt.figure(figsize=(12, 7))
        plt.scatter(rpm_np, torque_nm_np, label=f'Original Data Points (V ≈ {target_velocity_mph} mph)', color='blue', zorder=5)
        plt.plot(rpm_fit_line, torque_fit_line_rpm, label=f'Quadratic Regression (deg={degree_torque_rpm})', color='red', linestyle='--')
        plt.xlabel('Propeller RPM')
        plt.ylabel(f'Torque (N-m) at V ≈ {target_velocity_mph} mph')
        plt.title(f'Propeller Torque vs. RPM at V ≈ {target_velocity_mph} mph')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif len(rpms) > 0 :
        print("Not enough data points for quadratic Torque vs RPM regression. Plotting points only.")
        # ... (plotting code for points only if needed)
    else:
        print("No data to plot for Torque vs RPM.")


    # --- Plot 2: Thrust vs Torque ---
    if len(torque_nm_np) >= 3 and len(thrust_n_np) >= 3: # Need at least 3 points for quadratic regression
        degree_thrust_torque = 2 # Changed to quadratic
        coeffs_thrust_torque = np.polyfit(torque_nm_np, thrust_n_np, degree_thrust_torque)
        poly_func_thrust_torque = np.poly1d(coeffs_thrust_torque)
        print(f"\nRegression for Thrust vs Torque (Thrust = a*Torque^2 + b*Torque + c):")
        print(f"a = {coeffs_thrust_torque[0]:.4e}")
        print(f"b = {coeffs_thrust_torque[1]:.4f}")
        print(f"c = {coeffs_thrust_torque[2]:.4f}")
        print(f"Equation: Thrust_N ≈ ({coeffs_thrust_torque[0]:.3e})*Torque_Nm^2 + ({coeffs_thrust_torque[1]:.3f})*Torque_Nm + ({coeffs_thrust_torque[2]:.3f})")

        min_torque_for_fit = min(torque_nm_np)
        max_torque_for_fit = max(torque_nm_np)
        
        # Ensure linspace has a valid range
        if (max_torque_for_fit - min_torque_for_fit < 1e-9) :
             min_torque_for_fit -= 0.1 * abs(min_torque_for_fit) if abs(min_torque_for_fit)>1e-9 else 0.01
             max_torque_for_fit += 0.1 * abs(max_torque_for_fit) if abs(max_torque_for_fit)>1e-9 else 0.01
             if min_torque_for_fit == max_torque_for_fit:
                min_torque_for_fit -= 0.01
                max_torque_for_fit += 0.01
        
        torque_fit_line_thrust = np.linspace(min_torque_for_fit, max_torque_for_fit, 200)
        thrust_fit_line_torque = poly_func_thrust_torque(torque_fit_line_thrust)

        plt.figure(figsize=(12, 7))
        scatter = plt.scatter(torque_nm_np, thrust_n_np, c=rpm_np, cmap='viridis', s=50, label=f'Data (V ≈ {target_velocity_mph} mph)', zorder=5)
        plt.plot(torque_fit_line_thrust, thrust_fit_line_torque, label=f'Quadratic Regression (deg={degree_thrust_torque})', color='red', linestyle='--')
        
        for i, rpm_val in enumerate(rpm_np):
            plt.annotate(f"{rpm_val}", (torque_nm_np[i], thrust_n_np[i]),
                         textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

        cbar = plt.colorbar(scatter, label='RPM')
        plt.xlabel(f'Torque (N-m) at V ≈ {target_velocity_mph} mph')
        plt.ylabel(f'Thrust (N) at V ≈ {target_velocity_mph} mph')
        plt.title(f'Thrust vs. Torque at V ≈ {target_velocity_mph} mph (Color-coded by RPM)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif len(torque_nm_np) > 0 and len(thrust_n_np) > 0:
        print("Not enough data points for Thrust vs Torque quadratic regression. Plotting points only.")
        # ... (plotting code for points only if needed)
    else:
        print("No data to plot for Thrust vs Torque.")

if __name__ == '__main__':
    main()

#  r'C:\Users\kilianseibl\Downloads\PERFILES_WEB-202503\PERFILES_WEB\PERFILES2\PER3_4x4E-3.txt'