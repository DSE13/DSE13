# --- User Inputs ---
S_n_target_val = 0.0040       # m, Total BL thickness
h_1_val = 0.00009171         # m, First layer thickness
h_adj_cell_val = 0.0013      # m, Height of first adjacent cell

# Transition Ratios to test
# You can provide a single value or a list
# target_TR_options = 0.5
target_TR_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Test a range

# Optional: Maximum number of layers to iterate up to
max_n_check = 30 # Typically, inflation layers are not excessively numerous

# Optional: Tolerance for matching S_n_target (e.g., 1% = 1.0)
S_n_match_tolerance_percent = 0.5 # Try to get within 0.5% of the target S_n


def calculate_inflation_params(S_n_target, h_1, h_adj_cell, target_TR_values, n_max=50, S_n_tolerance_percent=1.0):
    """
    Calculates the number of inflation layers (n) and growth rate (r)
    to achieve a target total thickness (S_n_target) and first layer height (h_1),
    while respecting a transition ratio (TR) with the adjacent cell (h_adj_cell).

    Args:
        S_n_target (float): Desired total boundary layer thickness.
        h_1 (float): First layer thickness.
        h_adj_cell (float): Height of the first adjacent cell outside the BL.
        target_TR_values (list or float): A single TR value or a list of TR values to test.
                                          TR must be between 0 (exclusive) and 1 (inclusive).
        n_max (int): Maximum number of layers to check.
        S_n_tolerance_percent (float): Allowable percentage deviation for S_n_calc from S_n_target.

    Returns:
        list: A list of dictionaries, where each dictionary contains a valid
              set of parameters: {'TR', 'n', 'r', 'S_n_calculated', 'h_n_calculated'}.
    """
    if not isinstance(target_TR_values, list):
        target_TR_values = [target_TR_values]

    valid_solutions = []
    S_n_abs_tolerance = (S_n_tolerance_percent / 100.0) * S_n_target

    print(f"Target S_n: {S_n_target:.6f} m")
    print(f"First layer h_1: {h_1:.8f} m")
    print(f"Adjacent cell h_adj_cell: {h_adj_cell:.6f} m")
    print(f"Max layers to check (n_max): {n_max}")
    print(f"S_n tolerance: +/- {S_n_tolerance_percent}% ({S_n_abs_tolerance:.8f} m)\n")

    if h_1 >= S_n_target:
        print("Warning: First layer height h_1 is greater than or equal to target total thickness S_n_target.")
        print("This typically means n=1, and growth rate/TR are not applicable in the usual sense.")
        if abs(h_1 - S_n_target) < S_n_abs_tolerance: # Check if n=1 is the solution
            # For n=1, h_n = h_1. TR = h_1 / h_adj_cell
            # r is undefined/irrelevant for n=1
            tr_for_n1 = h_1 / h_adj_cell
            if 0 < tr_for_n1 <= 1.0 and any(abs(tr_for_n1 - tr_val) < 0.01 for tr_val in target_TR_values): # Check if this TR is desired
                 valid_solutions.append({
                    'TR': tr_for_n1,
                    'n': 1,
                    'r': float('nan'), # Growth rate is not well-defined for n=1
                    'S_n_calculated': h_1,
                    'h_n_calculated': h_1
                })
        return valid_solutions # Or handle as an error if n>1 is strictly required


    for TR in target_TR_values:
        if not (0 < TR <= 1.0):
            print(f"Skipping TR = {TR:.2f} as it's outside the valid range (0, 1].")
            continue

        print(f"--- Evaluating for Target Transition Ratio (TR) = {TR:.3f} ---")
        found_for_this_TR = False

        # n must be at least 2 for the formula r = ((TR * h_adj_cell) / h_1)^(1 / (n-1)) to be meaningful
        # and for a growth rate to apply.
        for n_candidate in range(2, n_max + 1):
            # Calculate required growth rate r for this n and TR
            # r^(n-1) = (TR * h_adj_cell) / h_1
            # This term must be > 0. Since TR>0, h_adj_cell>0, h_1>0, it will be.
            term_for_r_base = (TR * h_adj_cell) / h_1

            if term_for_r_base <= 0: # Should not happen with valid TR > 0
                continue

            # If term_for_r_base == 1, then r = 1.
            # If term_for_r_base < 1, then r < 1 (shrinkage, if n-1 is odd, or complex if even root of neg)
            #   Actually, (positive_num < 1) ^ (positive_exponent) is still < 1. So r < 1.
            # If term_for_r_base > 1, then r > 1 (growth)
            
            if n_candidate == 1: # Should be caught by loop start range(2,...)
                # This case means h_n = h_1, so TR * h_adj_cell = h_1.
                # S_n = h_1. Only valid if S_n_target == h_1.
                # r is undefined.
                continue # Skip, handled by pre-check or range of n

            try:
                r_candidate = term_for_r_base**(1 / (n_candidate - 1))
            except ZeroDivisionError: # Should not happen as n_candidate starts from 2
                continue
            
            # We are looking for growth, so r > 1.
            # If r_candidate is very slightly less than 1 (e.g. 0.99999999999) due to precision,
            # it might still be a valid "near constant" layer thickness.
            # However, Ansys typically expects r >= 1. Let's be strict: r > 1 for growth.
            # If r_candidate == 1.0, then h_n = h_1, and S_n = n_candidate * h_1
            # If r_candidate < 1.0, it's layer shrinkage, usually not desired for inflation.
            
            if r_candidate <= 1.000000001: # Check if r is not growing or shrinking too much
                                         # (use small epsilon for floating point comparisons if r=1 is allowed)
                                         # For strict growth, use r_candidate <= 1.0
                if abs(r_candidate - 1.0) < 1e-9: # r is effectively 1
                    S_n_calculated = n_candidate * h_1
                else: # r < 1, shrinkage
                    # print(f"  n={n_candidate}: r={r_candidate:.4f} (shrinkage/constant, skipping)")
                    continue # Skip if strictly seeking growth, or handle r=1 separately
            # else: # r_candidate > 1 (growth)
            #     S_n_calculated = h_1 * (r_candidate**n_candidate - 1) / (r_candidate - 1)
            
            # General formula for S_n (works for r=1 if handled by limit, but safer to separate)
            if abs(r_candidate - 1.0) < 1e-9: # r is effectively 1
                S_n_calculated = n_candidate * h_1
            else: # r != 1
                S_n_calculated = h_1 * (r_candidate**n_candidate - 1) / (r_candidate - 1)


            # Check if calculated S_n is within tolerance of target S_n
            if abs(S_n_calculated - S_n_target) <= S_n_abs_tolerance:
                h_n_calculated = h_1 * r_candidate**(n_candidate - 1)
                # Double check TR
                actual_TR = h_n_calculated / h_adj_cell
                
                # Ensure h_n is not excessively large or small causing issues
                if S_n_calculated > 0 and h_n_calculated > 0 and h_n_calculated < S_n_calculated:
                    solution = {
                        'TR_target': TR,
                        'n': n_candidate,
                        'r': r_candidate,
                        'S_n_calculated': S_n_calculated,
                        'h_n_calculated': h_n_calculated,
                        'Actual_TR_achieved': actual_TR
                    }
                    valid_solutions.append(solution)
                    found_for_this_TR = True
                    # print(f"  Found solution for TR={TR:.3f}: n={n_candidate}, r={r_candidate:.4f}, S_n_calc={S_n_calculated:.6f}, h_n_calc={h_n_calculated:.8f}, Actual TR={actual_TR:.3f}")


        if not found_for_this_TR:
            print(f"  No solution found for TR={TR:.3f} within S_n tolerance and n_max.")
        print("-" * 40)
    return valid_solutions

# --- Run Calculation ---
solutions = calculate_inflation_params(
    S_n_target_val,
    h_1_val,
    h_adj_cell_val,
    target_TR_options,
    n_max=max_n_check,
    S_n_tolerance_percent=S_n_match_tolerance_percent
)

# --- Output Results ---
if solutions:
    print("\n=== Found Valid Inflation Settings ===")
    for sol in solutions:
        print(
            f"Target TR: {sol['TR_target']:.3f} => "
            f"Num Layers (n): {sol['n']:2d}, "
            f"Growth Rate (r): {sol['r']:.4f}, "
            f"Total Thickness (S_n): {sol['S_n_calculated']:.6f} m, "
            f"Last Layer (h_n): {sol['h_n_calculated']:.8f} m, "
            f"Actual TR: {sol['Actual_TR_achieved']:.3f}"
        )
else:
    print("\nNo suitable inflation settings found for the given criteria.")

print("\n--- Notes ---")
print("1. 'Num Layers (n)' and 'Growth Rate (r)' are the primary inputs for Ansys.")
print("2. 'Transition Ratio (TR)' is typically set in Ansys (often called 'Maximum Size Ratio' or similar with adjacent cells, or it's an outcome of settings).")
print("3. The 'Actual TR Achieved' should be very close to your 'Target TR'.")
print(f"4. Total BL Thickness S_n should be close to {S_n_target_val:.6f} m (within +/- {S_n_match_tolerance_percent}%).")
print(f"5. First layer thickness h_1 is fixed at {h_1_val:.8f} m by these calculations.")