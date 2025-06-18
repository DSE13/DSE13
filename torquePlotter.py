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
    file_path = r'C:\Users\kilianseibl\Downloads\DSE13\PER3_41x41E.txt'
    m_s_to_mph = 2.236936
    m_s = 20
    target_velocity_mph = m_s_to_mph * m_s
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
    can_attempt_torque_regression = False
    rpm_reg_torque_np = np.array([])  # For storing RPMs used in torque regression
    torque_reg_nm_np = np.array([]) # For storing torques used in torque regression

    # Ensure we have initial data (filtered by thrust_threshold) before attempting to filter for torque regression
    if len(rpm_plot_np) > 0 and len(torque_plot_nm_np) > 0:
        min_significant_torque = 0.002  # Define a threshold for "significant" torque for regression (N·m)
        
        # Find indices where torque in the initially filtered data (by thrust) is significant
        significant_torque_indices = np.where(torque_plot_nm_np >= min_significant_torque)[0]

        if len(significant_torque_indices) > 0:
            # Use all points above the significance threshold for regression
            _rpm_reg_temp_torque = rpm_plot_np[significant_torque_indices]
            _torque_reg_temp_nm = torque_plot_nm_np[significant_torque_indices]

            if len(_rpm_reg_temp_torque) >= 3:  # Need at least 3 points for a quadratic fit
                rpm_reg_torque_np = _rpm_reg_temp_torque
                torque_reg_nm_np = _torque_reg_temp_nm
                can_attempt_torque_regression = True
            else:
                print(f"Not enough data points (found {len(_rpm_reg_temp_torque)} after filtering for Torque >= {min_significant_torque} N·m on data with Thrust <= {thrust_threshold}N) for quadratic Torque vs RPM regression.")
        else:
            print(f"No data points found with Torque >= {min_significant_torque} N·m (on data with Thrust <= {thrust_threshold}N) for Torque vs RPM regression.")
    
    if can_attempt_torque_regression:
        degree_torque_rpm = 2
        coeffs_torque_rpm = np.polyfit(rpm_reg_torque_np, torque_reg_nm_np, degree_torque_rpm)
        poly_func_torque_rpm = np.poly1d(coeffs_torque_rpm)
        
        print(rf"Regression for Torque vs RPM (using data subset where original Thrust $\leq$ {thrust_threshold}N and Torque $\geq$ {min_significant_torque} N·m):")
        print(rf"$Torque = a \cdot RPM^2 + b \cdot RPM + c$")
        print(f"a = {coeffs_torque_rpm[0]:.4e}")
        print(f"b = {coeffs_torque_rpm[1]:.4e}")
        print(f"c = {coeffs_torque_rpm[2]:.4e}")

        # Calculate R-squared on the regression subset
        y_pred_torque_rpm_reg = poly_func_torque_rpm(rpm_reg_torque_np)
        ss_res_torque_rpm = np.sum((torque_reg_nm_np - y_pred_torque_rpm_reg)**2)
        ss_tot_torque_rpm = np.sum((torque_reg_nm_np - np.mean(torque_reg_nm_np))**2)
        r_squared_torque_rpm = 1 - (ss_res_torque_rpm / ss_tot_torque_rpm) if ss_tot_torque_rpm > 0 else 0
        print(f"R-squared (on regression subset for Torque vs RPM): {r_squared_torque_rpm:.4f}")
        
        # Determine RPM range for plotting the fit line, based on the regression data
        min_rpm_for_fit_torque = min(rpm_reg_torque_np) if len(rpm_reg_torque_np) > 0 else 0
        max_rpm_for_fit_torque = max(rpm_reg_torque_np) if len(rpm_reg_torque_np) > 0 else 1
        
        if (max_rpm_for_fit_torque - min_rpm_for_fit_torque < 1e-9) : # Ensure a valid range for linspace
             min_rpm_for_fit_torque -= 0.1 * abs(min_rpm_for_fit_torque) if abs(min_rpm_for_fit_torque)>1e-9 else 0.01
             max_rpm_for_fit_torque += 0.1 * abs(max_rpm_for_fit_torque) if abs(max_rpm_for_fit_torque)>1e-9 else 0.01
             if abs(min_rpm_for_fit_torque - max_rpm_for_fit_torque) < 1e-9: # check if they are still effectively the same
                min_rpm_for_fit_torque -= 0.01 
                max_rpm_for_fit_torque += 0.01
        
        rpm_fit_line_torque_plot = np.linspace(min_rpm_for_fit_torque, max_rpm_for_fit_torque, 200)
        torque_fit_line_rpm_plot = poly_func_torque_rpm(rpm_fit_line_torque_plot)

        plt.figure(figsize = (my_width, my_width/golden))
        # Scatter plot ALL data points that met the initial thrust_threshold
        plt.scatter(rpm_plot_np, torque_plot_nm_np, label=rf'All Data (Thrust $\leq$ {thrust_threshold}N)', color='blue', zorder=5, s=20)
        # Plot the regression line based on the subset of data with significant torque
        plt.plot(rpm_fit_line_torque_plot, torque_fit_line_rpm_plot, label=rf'Quadr. Regr. (R$^2$={r_squared_torque_rpm:.3f})', color='red', linestyle='--')
        
        plt.xlabel('RPM [1/s]')
        plt.ylabel(rf'Torque [$N \cdot m$]')
        plt.title(f'V $\\approx$ {target_velocity_mps} m/s')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif len(rpm_plot_np) > 0 and len(torque_plot_nm_np) > 0: 
        # Fallback: Plot points only if regression couldn't be performed 
        # (e.g. not enough significant torque data or initial data)
        # but there was some data meeting the initial thrust_threshold.
        min_significant_torque = 0.01 # Define here as well for the print message
        print(f"Plotting points only for Torque vs RPM (Thrust <= {thrust_threshold}N). Regression on subset (Torque >= {min_significant_torque} N·m) failed or not enough data.")
        plt.figure(figsize = (my_width, my_width/golden))
        plt.scatter(rpm_plot_np, torque_plot_nm_np, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', color='blue', zorder=5, s=20)
        plt.xlabel('RPM [1/s]')
        plt.ylabel(rf'Torque [$N \cdot m$]')
        plt.title(rf'Torque vs. RPM (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s) - Points Only')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else: # No data at all met the initial thrust_threshold
        print(f"No data to plot for Torque vs RPM with Thrust <= {thrust_threshold}N.")

    # --- Plot 2: Thrust vs RPM (Filtered Data) ---
    can_attempt_regression = False
    rpm_reg_np = np.array([])
    thrust_reg_n_np = np.array([])

    # Ensure we have initial data before attempting to filter for regression
    if len(rpm_plot_np) > 0 and len(thrust_plot_n_np) > 0:
        min_significant_thrust = 0.05  # Threshold for "significant" thrust for regression
        
        # Find indices where thrust in the initially filtered data is significant
        significant_thrust_indices = np.where(thrust_plot_n_np >= min_significant_thrust)[0]

        if len(significant_thrust_indices) > 0:
            first_significant_idx = significant_thrust_indices[0]
            
            # Create a subset of data for regression (where thrust is significant)
            _rpm_reg_temp = rpm_plot_np[first_significant_idx:]
            _thrust_reg_temp = thrust_plot_n_np[first_significant_idx:]

            if len(_rpm_reg_temp) >= 3:  # Need at least 3 points for a quadratic fit
                rpm_reg_np = _rpm_reg_temp
                thrust_reg_n_np = _thrust_reg_temp
                can_attempt_regression = True
            else:
                print(f"Not enough data points (found {len(_rpm_reg_temp)} after filtering for Thrust >= {min_significant_thrust}N) for quadratic Thrust vs RPM regression.")
        else:
            print(f"No data points found with Thrust >= {min_significant_thrust}N for regression.")
    
    if can_attempt_regression:
        degree_thrust_rpm = 2 
        coeffs_thrust_rpm = np.polyfit(rpm_reg_np, thrust_reg_n_np, degree_thrust_rpm)
        poly_func_thrust_rpm = np.poly1d(coeffs_thrust_rpm)
        
        print(rf"Regression for Thrust vs RPM (using data subset where original Thrust $\geq$ {min_significant_thrust}N and $\leq$ {thrust_threshold}N):")
        print(rf"$Thrust = a \cdot RPM^2 + b \cdot RPM + c$")
        print(f"a = {coeffs_thrust_rpm[0]:.4e}")
        print(f"b = {coeffs_thrust_rpm[1]:.4e}")
        print(f"c = {coeffs_thrust_rpm[2]:.4e}")

        # Calculate R-squared on the regression subset
        y_pred_thrust_rpm_reg = poly_func_thrust_rpm(rpm_reg_np)
        ss_res_thrust_rpm = np.sum((thrust_reg_n_np - y_pred_thrust_rpm_reg)**2)
        ss_tot_thrust_rpm = np.sum((thrust_reg_n_np - np.mean(thrust_reg_n_np))**2)
        r_squared_thrust_rpm = 1 - (ss_res_thrust_rpm / ss_tot_thrust_rpm) if ss_tot_thrust_rpm > 0 else 0
        print(f"R-squared (on regression subset): {r_squared_thrust_rpm:.4f}")
        
        # Determine RPM range for plotting the fit line, based on the regression data
        min_rpm_for_fit = min(rpm_reg_np) if len(rpm_reg_np) > 0 else 0
        max_rpm_for_fit = max(rpm_reg_np) if len(rpm_reg_np) > 0 else 1 
        
        if (max_rpm_for_fit - min_rpm_for_fit < 1e-9) : # Ensure a valid range for linspace
             min_rpm_for_fit -= 0.1 * abs(min_rpm_for_fit) if abs(min_rpm_for_fit)>1e-9 else 0.01
             max_rpm_for_fit += 0.1 * abs(max_rpm_for_fit) if abs(max_rpm_for_fit)>1e-9 else 0.01
             if min_rpm_for_fit == max_rpm_for_fit: 
                min_rpm_for_fit -= 0.01 
                max_rpm_for_fit += 0.01
        
        rpm_fit_line_thrust = np.linspace(min_rpm_for_fit, max_rpm_for_fit, 200)
        thrust_fit_line_rpm = poly_func_thrust_rpm(rpm_fit_line_thrust)

        plt.figure(figsize = (my_width, my_width/golden))
        # Scatter plot ALL data points that met the initial thrust_threshold
        plt.scatter(rpm_plot_np, thrust_plot_n_np, label=rf'All Data (Thrust $\leq$ {thrust_threshold}N)', color='blue', zorder=5, s=20)
        # Plot the regression line based on the subset of data with significant thrust
        plt.plot(rpm_fit_line_thrust, thrust_fit_line_rpm, label=rf'Quadr. Regr. (R$^2$={r_squared_thrust_rpm:.3f})', color='red', linestyle='--')
        
        plt.xlabel(rf'RPM [1/s]') 
        plt.ylabel(rf'Thrust [N]')
        plt.title(f'V $\\approx$ {target_velocity_mps} m/s')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif len(rpm_plot_np) > 0 and len(thrust_plot_n_np) > 0: 
        # Fallback: Plot points only if regression couldn't be performed (e.g. not enough significant thrust data)
        # but there was some data meeting the initial thrust_threshold.
        print(f"Plotting points only for Thrust vs RPM (Thrust <= {thrust_threshold}N). Regression on subset (Thrust >= {min_significant_thrust}N) failed or not enough data.")
        plt.figure(figsize = (my_width, my_width/golden))
        scatter = plt.scatter(rpm_plot_np, thrust_plot_n_np, c=rpm_plot_np, cmap='viridis', s=50, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', zorder=5)
        cbar = plt.colorbar(scatter, label='RPM')
        plt.xlabel(rf'RPM [1/s]') 
        plt.ylabel(rf'Thrust [N]')
        plt.title(rf'Thrust vs. RPM (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s) - Points Only') 
        plt.legend() # legend() is called for consistency, though label might be the only item.
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else: # No data at all met the initial thrust_threshold
        print(f"No data to plot for Thrust vs RPM with Thrust <= {thrust_threshold}N.")


    # --- Plot 3: Power vs Thrust (Filtered Data) ---
    if len(thrust_plot_n_np) >= 3 and len(power_plot_w_np) >= 3: # Need at least 3 points for Quadr. Regr.
            degree_power_thrust = 2
            coeffs_power_thrust = np.polyfit(thrust_plot_n_np, power_plot_w_np, degree_power_thrust)
            poly_func_power_thrust = np.poly1d(coeffs_power_thrust)
            print(rf"Regression for Power vs Thrust (Thrust $\leq$ {thrust_threshold}N, $Power = a \cdot Thrust^2 + b \cdot Thrust + c$):")
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
            scatter = plt.scatter(thrust_plot_n_np, power_plot_w_np, color='blue', label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', zorder=5, s=20)
            plt.plot(thrust_fit_line_power, power_fit_line_thrust, label=rf'Quadr. Regr. (R$^2$={r_squared_power_thrust:.3f})', color='red', linestyle='--')
            
            #cbar = plt.colorbar(scatter, label='RPM')
            plt.xlabel(rf'Thrust [N]')
            plt.ylabel(rf'Power [W]')
            plt.title(f'V $\\approx$ {target_velocity_mps} m/s')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    elif len(thrust_plot_n_np) > 0:
        print(f"Not enough data points for Power vs Thrust Quadr. Regr. with Thrust <= {thrust_threshold}N. Plotting points only.")
        plt.figure(figsize = (my_width, my_width/golden))
        #scatter = plt.scatter(thrust_plot_n_np, power_plot_w_np, c=rpm_plot_np, cmap='viridis', s=50, label=rf'Data (Thrust $\leq$ {thrust_threshold}N)', zorder=5)
        #cbar = plt.colorbar(scatter, label='RPM')
        plt.xlabel(rf'Thrust [N]')
        plt.ylabel(rf'Power [W]')
        plt.title(rf'Power vs. Thrust (Thrust $\leq$ {thrust_threshold}N, V $\approx$ {target_velocity_mps} m/s) - Points Only')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data to plot for Power vs Thrust with Thrust <= {thrust_threshold}N.")

if __name__ == '__main__':
    main()

