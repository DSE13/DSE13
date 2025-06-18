import matplotlib.pyplot as plt

plt.style.use("paper.mplstyle")

pt = 1./72.27
jour_sizes = {"PRD": {"onecol": 483.69687*pt, "halfcol": 241.84843*pt, "quartercol": 120.92421*pt}}
my_width = jour_sizes["PRD"]["halfcol"]
golden = (1 + 5**0.5) / 2

# --- Manually Input Data ---

x_loc = [
    9.88E-02, 9.72E-02, 9.52E-02, 9.28E-02, 8.97E-02, 8.60E-02,
    8.15E-02, 7.63E-02, 7.03E-02, 6.37E-02, 5.65E-02, 4.91E-02,
    4.16E-02, 3.43E-02, 2.76E-02, 2.15E-02, 1.63E-02, 1.18E-02,
    8.20E-03, 5.30E-03, 3.10E-03, 1.40E-03, 4.00E-04
]
###AOA 0###
cp_lower = [
    3.42E-01, 1.68E-01, -2.87E-02, -1.96E-01, -3.14E-01, -3.76E-01,
    -3.89E-01, -3.63E-01, -3.04E-01, -2.18E-01, -1.12E-01, 4.10E-03,
    1.20E-01, 2.28E-01, 3.16E-01, 3.77E-01, 4.07E-01, 3.98E-01,
    3.40E-01, 2.26E-01, 6.17E-02, -1.36E-01, -3.21E-01
]

cp_upper = [
    -3.42E-01, -1.68E-01, 2.87E-02, 1.96E-01, 3.14E-01, 3.76E-01,
    3.89E-01, 3.63E-01, 3.04E-01, 2.18E-01, 1.12E-01, -4.10E-03,
    -1.20E-01, -2.28E-01, -3.16E-01, -3.77E-01, -4.07E-01, -3.98E-01,
    -3.40E-01, -2.26E-01, -6.18E-02, 1.37E-01, 3.21E-01 # Note: some values are not just negative of cp_lower
]

# ##AOA 20###
# cp_lower =  [
#     3.00E-04,
#     6.00E-04,
#     6.00E-04,
#     6.00E-04,
#     1.00E-03,
#     1.40E-03,
#     2.00E-03,
#     2.60E-03,
#     3.30E-03,
#     4.20E-03,
#     5.40E-03,
#     6.60E-03,
#     8.20E-03,
#     1.02E-02,
#     1.24E-02,
#     1.50E-02,
#     1.78E-02,
#     2.10E-02,
#     2.42E-02,
#     2.66E-02,
#     2.72E-02,
#     2.44E-02,
#     1.70E-02
# ]

# cp_upper = [
#     4.80E-03,
#     1.56E-02,
#     7.60E-03,
#     -5.40E-03,
#     -1.44E-02,
#     -1.76E-02,
#     -1.91E-02,
#     -2.14E-02,
#     -2.39E-02,
#     -2.64E-02,
#     -2.87E-02,
#     -3.10E-02,
#     -3.11E-02,
#     -3.18E-02,
#     -2.89E-02,
#     -2.11E-02,
#     -7.90E-03,
#     1.06E-02,
#     3.32E-02,
#     5.69E-02,
#     7.57E-02,
#     8.09E-02,
#     6.47E-02 
# ]


# --- Sanity Check: Ensure all lists have the same length ---
if not (len(x_loc) == len(cp_lower) == len(cp_upper)):
    print("Error: Data lists have different lengths!")
    print(f"Length of x_loc: {len(x_loc)}")
    print(f"Length of cp_lower: {len(cp_lower)}")
    print(f"Length of cp_upper: {len(cp_upper)}")
    exit()

# --- Create Plot ---
plt.figure(figsize=(my_width, my_width / golden)) # Use defined width and golden ratio

# Plot Lower Wing CP
plt.plot(x_loc, cp_lower, marker='o', linestyle='-', label=r'L. Wing $\Delta C_P$ (Prop. - No Prop.)', markersize=4)

# Plot Upper Wing CP
plt.plot(x_loc, cp_upper, marker='x', linestyle='-', label=r'U. Wing $\Delta C_P$ (Prop. - No Prop.)', markersize=4)

# --- Customize Plot ---
plt.title(r'AoA = $0^{\circ}$')
plt.xlabel(r'$x/c$')
plt.ylabel(r'$\Delta C_P$')
#plt.legend() # Show legend to identify lines
# plt.grid(True) # Grid is handled by paper.mplstyle

# For typical airfoil pressure plots, the y-axis is often inverted
plt.gca().invert_yaxis()

# plt.tight_layout() # Handled by figure.autolayout=True in paper.mplstyle

# --- Show Plot ---
plt.show()