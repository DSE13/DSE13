import matplotlib.pyplot as plt
import numpy as np

# Data (transcribed from your input)
# Note: I've removed the initial coefficient name from the data strings
cd_tot_str = "-4.718273939900000102e-02,1.149351997590000035e-01,-1.129406275389999992e-01,-7.387942270899999775e-02,2.175714780570000006e-01,1.295935577210000078e-01,1.833637520660000020e-01,6.317806504600000095e-02,7.987754383900000366e-02,-4.000346803460600142e+01,-4.891706543000000004e-03,-4.329861807828499991e+02,-5.832418417199999872e-02,1.496340213619999915e-01,-3.773006353600000251e-02,1.189296889430000032e-01,9.653857296699999702e-02,3.486427718369999873e-01,1.723091328159999935e-01,1.149687756740000039e-01,9.169139616400000248e-02,-5.802374090399999712e-02,1.147556908830000000e-01,5.466193068599999733e-02,4.835093415999999582e-03"
cl_str = "1.092168049033999999e+00,1.089409858578000101e+00,1.333331470824000053e+00,1.070648313086000059e+00,1.177470085131999911e+00,1.004453915218999960e+00,1.044162900172999953e+00,1.047786412757000019e+00,9.535904764870000161e-01,3.711633429244299975e+01,1.544837044417999961e+00,-1.513575377765079111e+03,1.073308727046000000e+00,9.593777891620000053e-01,1.023264928911000071e+00,1.076220205350999937e+00,1.118689645172000091e+00,1.376986168806999977e+00,9.587647243600000424e-01,1.004141764300999995e+00,9.012589032400000200e-01,1.374075423552999942e+00,1.234671447248000042e+00,1.129428610219999962e+00,9.992671603849999640e-01"
wake_iter_str = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"

# Convert string data to lists of floats/integers
cd_tot = [float(x) for x in cd_tot_str.split(',')]
cl = [float(x) for x in cl_str.split(',')]
wake_iter = [int(x) for x in wake_iter_str.split(',')]

# --- Data Cleaning (Optional but Recommended) ---
# Identify and handle potential outliers if necessary.
# For this example, we'll plot as is, but in a real scenario,
# the large jumps in iteration 10 for CD and 12 for CL might warrant investigation.
# For example, you might want to cap the y-axis or remove outliers for better visualization
# of the general trend.

# --- Create the Plot ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot CDtot on the first y-axis (left)
color_cd = 'tab:red'
ax1.set_xlabel('Wake Iterations', fontsize=14)
ax1.set_ylabel('CDtot', color=color_cd, fontsize=14)
ax1.plot(wake_iter, cd_tot, color=color_cd, marker='o', linestyle='-', label='CDtot')
ax1.tick_params(axis='y', labelcolor=color_cd)
ax1.grid(True, linestyle='--', alpha=0.7)

# Create a second y-axis that shares the same x-axis for CL
ax2 = ax1.twinx()
color_cl = 'tab:blue'
ax2.set_ylabel('CL', color=color_cl, fontsize=14)
ax2.plot(wake_iter, cl, color=color_cl, marker='s', linestyle='--', label='CL')
ax2.tick_params(axis='y', labelcolor=color_cl)

# --- Add Plot Enhancements ---
plt.title('Convergence of CL and CDtot vs. Wake Iterations', fontsize=16)

# Add a combined legend
# To do this, get handles and labels from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)
# Could also use ax1.legend(...) if preferred, or fig.legend(...) for figure-level

# Set x-axis ticks to show all integer iterations if appropriate
if len(wake_iter) <= 30: # Only if not too many iterations, to avoid clutter
    ax1.set_xticks(wake_iter)
else:
    ax1.set_xticks(np.arange(min(wake_iter), max(wake_iter)+1, step=max(1, len(wake_iter)//10))) # Or some other intelligent step

# Optional: Adjust y-axis limits if outliers skew the plot too much
# Consider the range of your data. For example, if most CDtot values are small
# but one is very large, you might want to limit the y-axis for CDtot.
# Example (uncomment and adjust if needed):
ax1.set_ylim(-0.5, 0.5) # For CDtot if outliers are extreme
ax2.set_ylim(0, 2)    # For CL if outliers are extreme

plt.tight_layout() # Adjust layout to make room for labels
plt.show()