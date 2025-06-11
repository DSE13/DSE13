import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("paper.mplstyle")

pt = 1./72.27

jour_sizes = {"PRD": {"onecol": 483.69687*pt, "halfcol": 241.84843*pt, "quartercol": 120.92421*pt},
             }

my_width = jour_sizes["PRD"]["halfcol"]
golden = (1 + 5 ** 0.5) / 2

# Conversion factors
INLBF_TO_NM = 0.112984829
LBF_TO_N = 4.4482216153

def parse_prop_data(file_path, target_v_mph=33.0):
    """
    Parses the propeller data file to extract RPM, Torque (In-Lbf), Thrust (Lbf),
    Advance Ratio (J), Thrust Coefficient (Ct), and Power Coefficient (Cp)
    for the velocity closest to target_v_mph for each RPM section.

    Args:
        file_path (str): Path to the data file.
        target_v_mph (float): The target velocity in MPH.

    Returns:
        list: A list of tuples, where each tuple is
              (rpm, torque_in_lbf, thrust_lbf, actual_v_mph_for_data, j, ct, cp).
    """
    collected_data = []
    current_rpm = None
    min_v_diff_for_current_rpm = float('inf')
    best_torque_for_current_rpm = None
    best_thrust_for_current_rpm = None
    actual_v_at_best_data = None
    best_j_for_current_rpm = None
    best_ct_for_current_rpm = None
    best_cp_for_current_rpm = None

    rpm_regex = re.compile(r"^\s*PROP RPM =\s*(\d+)")

    with open(file_path, 'r') as f:
        for line in f:
            rpm_match = rpm_regex.match(line)
            if rpm_match:
                if current_rpm is not None and best_torque_for_current_rpm is not None \
                   and best_thrust_for_current_rpm is not None and best_j_for_current_rpm is not None \
                   and best_ct_for_current_rpm is not None and best_cp_for_current_rpm is not None:
                    collected_data.append((current_rpm, best_torque_for_current_rpm,
                                           best_thrust_for_current_rpm, actual_v_at_best_data,
                                           best_j_for_current_rpm, best_ct_for_current_rpm, best_cp_for_current_rpm))

                current_rpm = int(rpm_match.group(1))
                min_v_diff_for_current_rpm = float('inf')
                best_torque_for_current_rpm = None
                best_thrust_for_current_rpm = None
                actual_v_at_best_data = None
                best_j_for_current_rpm = None
                best_ct_for_current_rpm = None
                best_cp_for_current_rpm = None
                continue

            if current_rpm is None:
                continue

            parts = line.strip().split()
            # Ensure enough parts for J, Ct, Cp as well (e.g., at least 8 for original, plus more for new ones)
            # Adjust this condition based on the actual number of columns in your file
            if len(parts) >= 8: # Assuming J, Ct, Cp are within these first columns or require more
                try:
                    v_mph = float(parts[0])
                    # !!! Placeholder column indices - PLEASE VERIFY AND UPDATE !!!
                    j_val = float(parts[1])  # Assuming J is in column 1
                    ct_val = float(parts[2]) # Assuming Ct is in column 2
                    cp_val = float(parts[3]) # Assuming Cp is in column 3
                    torque_in_lbf = float(parts[6]) # Torque(In-Lbf) is 7th column
                    thrust_lbf = float(parts[7])    # Thrust(Lbf) is 8th column

                    v_diff = abs(v_mph - target_v_mph)

                    if v_diff < min_v_diff_for_current_rpm:
                        min_v_diff_for_current_rpm = v_diff
                        best_torque_for_current_rpm = torque_in_lbf
                        best_thrust_for_current_rpm = thrust_lbf
                        actual_v_at_best_data = v_mph
                        best_j_for_current_rpm = j_val
                        best_ct_for_current_rpm = ct_val
                        best_cp_for_current_rpm = cp_val
                except ValueError:
                    # Could occur if J, Ct, Cp columns are not numeric or parts[1,2,3] are out of bounds
                    pass
                except IndexError:
                    # Could occur if J, Ct, Cp columns are not present for a line
                    pass
        # Append last collected data if any
        if current_rpm is not None and best_torque_for_current_rpm is not None \
           and best_thrust_for_current_rpm is not None and best_j_for_current_rpm is not None \
           and best_ct_for_current_rpm is not None and best_cp_for_current_rpm is not None:
            collected_data.append((current_rpm, best_torque_for_current_rpm,
                                   best_thrust_for_current_rpm, actual_v_at_best_data,
                                   best_j_for_current_rpm, best_ct_for_current_rpm, best_cp_for_current_rpm))
    return collected_data


def main():
    file_path = r'C:\Users\kilianseibl\Downloads\PERFILES_WEB-202503\PERFILES_WEB\PERFILES2\PER3_4x45E.txt'
    target_velocity_mph = 35.79098
    target_velocity_mps = round(target_velocity_mph * 0.44704, 0)

    extracted_points = parse_prop_data(file_path, target_velocity_mph)

    if not extracted_points:
        print(f"No data points found near {target_velocity_mph} mph.")
        return

    rpms = []
    torques_nm = []
    thrusts_n = []
    powers_w = []
    adv_ratios_j = [] # New list for Advance Ratio (J)
    cts_coeff = []    # New list for Thrust Coefficient (Ct)
    cps_coeff = []    # New list for Power Coefficient (Cp)


    print(f"Extracted Data closest to V = {target_velocity_mps} m/s (All points):")
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print("RPM    | Actual V (mph) | J        | Ct       | Cp       | Torque (In-Lbf) | Torque (N-m) | Thrust (Lbf) | Thrust (N)  | Power (W)")
    print("-------|----------------|----------|----------|----------|-----------------|--------------|--------------|-------------|----------")
    for rpm, torque_in_lbf, thrust_lbf, actual_v, j_val, ct_val, cp_val in extracted_points:
        torque_newton_meter = torque_in_lbf * INLBF_TO_NM
        thrust_newton = thrust_lbf * LBF_TO_N
        power_watt = torque_newton_meter * rpm * (2 * np.pi / 60)

        rpms.append(rpm)
        torques_nm.append(torque_newton_meter)
        thrusts_n.append(thrust_newton)
        powers_w.append(power_watt)
        adv_ratios_j.append(j_val)
        cts_coeff.append(ct_val)
        cps_coeff.append(cp_val)

        print(f"{rpm:<7}| {actual_v:<14.2f} | {j_val:<8.4f} | {ct_val:<8.4f} | {cp_val:<8.4f} | {torque_in_lbf:<15.3f} | {torque_newton_meter:<12.4f} | {thrust_lbf:<12.3f} | {thrust_newton:<11.3f} | {power_watt:.2f}")
    print("-------------------------------------------------------------------------------------------------------------------------------")

    rpm_np = np.array(rpms)

    rpm_np = np.array(rpms)
    torque_nm_np = np.array(torques_nm)
    thrust_n_np = np.array(thrusts_n)
    power_w_np = np.array(powers_w)

    # --- Filter data for Thrust <= 2N for plots and regressions ---
    thrust_threshold = 2.0
    filter_indices = np.where(thrust_n_np <= thrust_threshold)[0]

    if len(filter_indices) == 0:
        print(f"\nNo data points found with Thrust <= {thrust_threshold}N. Skipping plots and regressions based on filtered data.")
        rpm_plot_np = np.array([])
        torque_plot_nm_np = np.array([])
        thrust_plot_n_np = np.array([])
        power_plot_w_np = np.array([])
    else:
        rpm_plot_np = rpm_np[filter_indices]
        torque_plot_nm_np = torque_nm_np[filter_indices]
        thrust_plot_n_np = thrust_n_np[filter_indices]
        power_plot_w_np = power_w_np[filter_indices]
        print(f"\nApplied filter: Thrust <= {thrust_threshold}N. Using {len(thrust_plot_n_np)} data points for subsequent plots and regressions.")

    # --- Plot 1: Torque vs RPM (Filtered Data) ---
    if len(rpm_plot_np) >= 3 and len(torque_plot_nm_np) >=3: # Need at least 3 points for quadratic
        degree_torque_rpm = 2
        coeffs_torque_rpm = np.polyfit(rpm_plot_np, torque_plot_nm_np, degree_torque_rpm)
        poly_func_torque_rpm = np.poly1d(coeffs_torque_rpm)
        print(rf"\nRegression for Torque vs RPM (Thrust $\leq$ {thrust_threshold}N, $Torque = a \cdot RPM^2 + b \cdot RPM + c$):")
        print(f"a = {coeffs_torque_rpm[0]:.4e}")
        print(f"b = {coeffs_torque_rpm[1]:.4e}")
        print(f"c = {coeffs_torque_rpm[2]:.4e}")
        
        # Calculate R-squared for Torque vs RPM
        y_pred_torque_rpm = poly_func_torque_rpm(rpm_plot_np)
        ss_res_torque_rpm = np.sum((torque_plot_nm_np - y_pred_torque_rpm)**2)
        ss_tot_torque_rpm = np.sum((torque_plot_nm_np - np.mean(torque_plot_nm_np))**2)
        r_squared_torque_rpm = 1 - (ss_res_torque_rpm / ss_tot_torque_rpm) if ss_tot_torque_rpm > 0 else 0
        print(f"R-squared (Torque vs RPM): {r_squared_torque_rpm:.4f}")

        rpm_fit_line = np.linspace(min(rpm_plot_np), max(rpm_plot_np), 200)
        torque_fit_line_rpm = poly_func_torque_rpm(rpm_fit_line)

        plt.figure(figsize = (my_width, my_width/golden))
        plt.scatter(rpm_plot_np, torque_plot_nm_np, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', color='blue', zorder=5)
        plt.plot(rpm_fit_line, torque_fit_line_rpm, label=rf'Quadratic Regression (R$^2$={r_squared_torque_rpm:.3f})', color='red', linestyle='--')
        plt.xlabel('RPM [1/s]')
        plt.ylabel(rf'Torque [$N \cdot m$]')
        plt.title(rf'Propeller Torque vs. RPM (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif len(rpm_plot_np) > 0:
        print(f"Not enough data points (found {len(rpm_plot_np)}) for quadratic Torque vs RPM regression with Thrust <= {thrust_threshold}N. Plotting points only.")
        plt.figure(figsize = (my_width, my_width/golden))
        plt.scatter(rpm_plot_np, torque_plot_nm_np, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', color='blue', zorder=5)
        plt.xlabel('RPM [1/s]')
        plt.ylabel(rf'Torque [$N \cdot m$]')
        plt.title(rf'Propeller Torque vs. RPM (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s) - Points Only')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data to plot for Torque vs RPM with Thrust <= {thrust_threshold}N.")


    # --- Plot 2: Thrust vs RPM (Filtered Data) ---
    if len(rpm_plot_np) >= 3 and len(thrust_plot_n_np) >= 3: # Need at least 3 points for quadratic regression
        degree_thrust_rpm = 2 
        coeffs_thrust_rpm = np.polyfit(rpm_plot_np, thrust_plot_n_np, degree_thrust_rpm)
        poly_func_thrust_rpm = np.poly1d(coeffs_thrust_rpm)
        print(rf"\nRegression for Thrust vs RPM (Thrust $\leq$ {thrust_threshold}N, $Thrust = a \cdot RPM^2 + b \cdot RPM + c$):")
        print(f"a = {coeffs_thrust_rpm[0]:.4e}")
        print(f"b = {coeffs_thrust_rpm[1]:.4f}")
        print(f"c = {coeffs_thrust_rpm[2]:.4f}")

        # Calculate R-squared for Thrust vs RPM
        y_pred_thrust_rpm = poly_func_thrust_rpm(rpm_plot_np)
        ss_res_thrust_rpm = np.sum((thrust_plot_n_np - y_pred_thrust_rpm)**2)
        ss_tot_thrust_rpm = np.sum((thrust_plot_n_np - np.mean(thrust_plot_n_np))**2)
        r_squared_thrust_rpm = 1 - (ss_res_thrust_rpm / ss_tot_thrust_rpm) if ss_tot_thrust_rpm > 0 else 0
        print(f"R-squared (Thrust vs RPM): {r_squared_thrust_rpm:.4f}")
        
        min_rpm_for_fit = min(rpm_plot_np)
        max_rpm_for_fit = max(rpm_plot_np)
        
        # Ensure linspace has a valid range (though less likely needed for RPM than for torque/thrust if data is sparse)
        if (max_rpm_for_fit - min_rpm_for_fit < 1e-9) : # Check for very small or zero range
             min_rpm_for_fit -= 0.1 * abs(min_rpm_for_fit) if abs(min_rpm_for_fit)>1e-9 else 0.01
             max_rpm_for_fit += 0.1 * abs(max_rpm_for_fit) if abs(max_rpm_for_fit)>1e-9 else 0.01
             if min_rpm_for_fit == max_rpm_for_fit: # Still equal after adjustment
                min_rpm_for_fit -= 0.01 # Fallback for zero original range
                max_rpm_for_fit += 0.01
        
        rpm_fit_line_thrust = np.linspace(min_rpm_for_fit, max_rpm_for_fit, 200)
        thrust_fit_line_rpm = poly_func_thrust_rpm(rpm_fit_line_thrust) # Using rpm_fit_line_thrust as input

        plt.figure(figsize = (my_width, my_width/golden))
        # Scatter plot x-axis is rpm_plot_np, color is also rpm_plot_np
        scatter = plt.scatter(rpm_plot_np, thrust_plot_n_np, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', color='blue', zorder=5)
        plt.plot(rpm_fit_line_thrust, thrust_fit_line_rpm, label=rf'Quadratic Regression (R$^2$={r_squared_thrust_rpm:.3f})', color='red', linestyle='--')
        
        plt.xlabel(rf'RPM [1/s]') # Changed X-axis label
        plt.ylabel(rf'Thrust [N]')
        plt.title(rf'APC B4x4.5E-B4 Thrust vs. RPM (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s)') # Updated title
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif len(rpm_plot_np) > 0 and len(thrust_plot_n_np) > 0: # Check both arrays have data
        print(f"Not enough data points for Thrust vs RPM quadratic regression with Thrust <= {thrust_threshold}N. Plotting points only.")
        plt.figure(figsize = (my_width, my_width/golden))
        scatter = plt.scatter(rpm_plot_np, thrust_plot_n_np, c=rpm_plot_np, cmap='viridis', s=50, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', zorder=5)
        cbar = plt.colorbar(scatter, label='RPM')
        plt.xlabel(rf'RPM [1/s]') # Changed X-axis label
        plt.ylabel(rf'Thrust [N]')
        plt.title(rf'APC B4x4.5E-B4 Thrust vs. RPM (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s) - Points Only') # Updated title
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data to plot for Thrust vs RPM with Thrust <= {thrust_threshold}N.")


    # --- Plot 3: Power vs Thrust (Filtered Data) ---
    if len(thrust_plot_n_np) >= 3 and len(power_plot_w_np) >= 3: # Need at least 3 points for quadratic regression
            degree_power_thrust = 2
            coeffs_power_thrust = np.polyfit(thrust_plot_n_np, power_plot_w_np, degree_power_thrust)
            poly_func_power_thrust = np.poly1d(coeffs_power_thrust)
            print(rf"\nRegression for Power vs Thrust (Thrust $\leq$ {thrust_threshold}N, $Power = a \cdot Thrust^2 + b \cdot Thrust + c$):")
            print(f"a = {coeffs_power_thrust[0]:.4e}")
            print(f"b = {coeffs_power_thrust[1]:.4f}")
            print(f"c = {coeffs_power_thrust[2]:.4f}")

            # Calculate R-squared for Power vs Thrust
            y_pred_power_thrust = poly_func_power_thrust(thrust_plot_n_np)
            ss_res_power_thrust = np.sum((power_plot_w_np - y_pred_power_thrust)**2)
            ss_tot_power_thrust = np.sum((power_plot_w_np - np.mean(power_plot_w_np))**2)
            r_squared_power_thrust = 1 - (ss_res_power_thrust / ss_tot_power_thrust) if ss_tot_power_thrust > 0 else 0
            print(f"R-squared (Power vs Thrust): {r_squared_power_thrust:.4f}")

            min_thrust_for_fit = min(thrust_plot_n_np)
            max_thrust_for_fit = max(thrust_plot_n_np)
            
            if (max_thrust_for_fit - min_thrust_for_fit < 1e-9) :
                min_thrust_for_fit -= 0.1 * abs(min_thrust_for_fit) if abs(min_thrust_for_fit)>1e-9 else 0.01
                max_thrust_for_fit += 0.1 * abs(max_thrust_for_fit) if abs(max_thrust_for_fit)>1e-9 else 0.01
                if min_thrust_for_fit == max_thrust_for_fit: 
                    min_thrust_for_fit -= 0.01
                    max_thrust_for_fit += 0.01
            
            thrust_fit_line_power = np.linspace(min_thrust_for_fit, max_thrust_for_fit, 200)
            power_fit_line_thrust = poly_func_power_thrust(thrust_fit_line_power)

            plt.figure(figsize = (my_width, my_width/golden))
            scatter = plt.scatter(thrust_plot_n_np, power_plot_w_np, c=rpm_plot_np, cmap='viridis', s=50, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', zorder=5)
            plt.plot(thrust_fit_line_power, power_fit_line_thrust, label=rf'Quadratic Regression (R$^2$={r_squared_power_thrust:.3f})', color='red', linestyle='--')
            
            cbar = plt.colorbar(scatter, label='RPM')
            plt.xlabel(rf'Thrust [N]')
            plt.ylabel(rf'Power [W]')
            plt.title(rf'APC B4x4.5E-B4 Power vs. Thrust (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    elif len(thrust_plot_n_np) > 0:
        print(f"Not enough data points for Power vs Thrust quadratic regression with Thrust <= {thrust_threshold}N. Plotting points only.")
        plt.figure(figsize = (my_width, my_width/golden))
        scatter = plt.scatter(thrust_plot_n_np, power_plot_w_np, c=rpm_plot_np, cmap='viridis', s=50, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', zorder=5)
        cbar = plt.colorbar(scatter, label='RPM')
        plt.xlabel(rf'Thrust [N]')
        plt.ylabel(rf'Power [W]')
        plt.title(rf'APC B4x4.5E-B4 Power vs. Thrust (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s) - Points Only')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data to plot for Power vs Thrust with Thrust <= {thrust_threshold}N.")

if __name__ == '__main__':
    main()

#  r'C:\Users\kilianseibl\Downloads\PERFILES_WEB-202503\PERFILES_WEB\PERFILES2\PER3_4x4E-3.txt'