import pandas as pd
import matplotlib.pyplot as plt
import os

def read_polar_file(file_path):
    try:
        # Read the file, treating the first line as the header and separating by whitespace
        df = pd.read_csv(file_path, delim_whitespace=True, header=0)
        print(f"Successfully read data from: {file_path}")
        print("Columns found:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty or has no data after skipping comments.")
        print("Please inspect the file manually.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def plot_polar_data(df, x_column, y_column, file_basename="Polar File"):
    """
    Plots data from the DataFrame.
    """
    if x_column not in df.columns:
        print(f"Error: X-axis column '{x_column}' not found in the data.")
        return
    if y_column not in df.columns:
        print(f"Error: Y-axis column '{y_column}' not found in the data.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column], marker='o', linestyle='-', label=f'{y_column} vs {x_column}')

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} vs {x_column} from Polar File')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiple_polar_data(dataframes_dict, x_column_name, y_column_name, title_prefix="Polar Data Comparison"):
    """
    Plots a specific X vs Y column from one or more pandas DataFrames on the same axes.
    dataframes_dict: A dictionary where keys are labels (e.g., filenames)
                     and values are the DataFrames.
    x_column_name: The name of the column to use for the X-axis.
    y_column_name: The name of the column to use for the Y-axis.
    """
    if not dataframes_dict:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 7)) # Adjusted figure size
    plotted_something = False

    for label, df in dataframes_dict.items():
        if df is not None and not df.empty:
            # Check if the requested columns exist in this specific DataFrame
            actual_x_col = None
            if x_column_name.lower() == 'aoa': # Handle common variations for AoA
                if 'Alpha' in df.columns: actual_x_col = 'Alpha'
                elif 'AoA' in df.columns: actual_x_col = 'AoA'
            elif x_column_name in df.columns:
                actual_x_col = x_column_name

            actual_y_col = None
            if y_column_name in df.columns:
                actual_y_col = y_column_name

            if actual_x_col and actual_y_col:
                plt.plot(df[actual_x_col], df[actual_y_col], marker='o', linestyle='-', label=label)
                plotted_something = True
            else:
                print(f"Skipping plot for '{label}':")
                if not actual_x_col: print(f"  X-column '{x_column_name}' (or AoA/Alpha) not found.")
                if not actual_y_col: print(f"  Y-column '{y_column_name}' not found.")
                # print(f"  Available columns in '{label}': {df.columns.tolist()}") # For debugging
        else:
            print(f"Skipping plot for '{label}' due to missing data.")

    if plotted_something:
        plt.xlabel(x_column_name) # Use the user's requested X name for the label
        plt.ylabel(y_column_name)
        plt.title(f"{title_prefix}: {y_column_name} vs {x_column_name}")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data could be plotted with the specified columns.")


if __name__ == "__main__":
    files_to_plot = []
    print("Enter the full paths to the .polar files you want to plot.")
    print("Press Enter after each file path. Press Enter on an empty line to finish.")

    while True:
        file_path = input(".polar file path (or press Enter to finish): ").strip()
        if not file_path:
            break
        if os.path.exists(file_path):
            files_to_plot.append(file_path)
        else:
            print(f"Warning: File '{file_path}' not found. It will be skipped.")

    if not files_to_plot:
        print("No valid files entered. Exiting.")
    else:
        data_to_display = {}
        first_df_columns = None # To store columns from the first successfully read file

        for f_path in files_to_plot:
            df = read_polar_file(f_path)
            if df is not None and not df.empty:
                file_label = os.path.splitext(os.path.basename(f_path))[0]
                data_to_display[file_label] = df
                if first_df_columns is None: # Get columns from the first file for user prompts
                    first_df_columns = df.columns.tolist()
            elif df is not None and df.empty:
                 print(f"File '{f_path}' was read but appears to be empty.")


        if data_to_display and first_df_columns:
            print("\n--- Select Columns for Plotting ---")
            print("Available columns (from the first successfully read file):")
            for i, col in enumerate(first_df_columns):
                print(f"  {i+1}. {col}")

            # Get X-axis column from user
            x_axis_col_chosen = ""
            x_axis_valid = False
            default_x_col = 'Alpha' if 'Alpha' in first_df_columns else ('AoA' if 'AoA' in first_df_columns else (first_df_columns[0] if first_df_columns else None))

            while not x_axis_valid:
                prompt_x = f"Enter the NAME or NUMBER for the X-AXIS (default: {default_x_col if default_x_col else 'None'}): "
                x_choice_str = input(prompt_x).strip()
                if not x_choice_str and default_x_col:
                    x_axis_col_chosen = default_x_col; x_axis_valid = True; print(f"Using default X-axis: {x_axis_col_chosen}")
                elif x_choice_str.isdigit():
                    try: x_idx = int(x_choice_str) - 1
                    except ValueError: print("Invalid number."); continue
                    if 0 <= x_idx < len(first_df_columns): x_axis_col_chosen = first_df_columns[x_idx]; x_axis_valid = True
                    else: print("Invalid number choice.")
                elif x_choice_str in first_df_columns: x_axis_col_chosen = x_choice_str; x_axis_valid = True
                else: print(f"Column '{x_choice_str}' not found.")

            # Get Y-axis column from user
            y_axis_col_chosen = ""
            y_axis_valid = False
            if x_axis_valid:
                default_y_candidates = [col for col in first_df_columns if col != x_axis_col_chosen]
                default_y_col = default_y_candidates[0] if default_y_candidates else (first_df_columns[0] if first_df_columns else None)
                while not y_axis_valid:
                    prompt_y = f"Enter NAME or NUMBER for Y-AXIS (vs '{x_axis_col_chosen}', default: {default_y_col if default_y_col else 'None'}): "
                    y_choice_str = input(prompt_y).strip()
                    if not y_choice_str and default_y_col:
                        y_axis_col_chosen = default_y_col; y_axis_valid = True; print(f"Using default Y-axis: {y_axis_col_chosen}")
                    elif y_choice_str.isdigit():
                        try: y_idx = int(y_choice_str) - 1
                        except ValueError: print("Invalid number."); continue
                        if 0 <= y_idx < len(first_df_columns): y_axis_col_chosen = first_df_columns[y_idx]; y_axis_valid = True
                        else: print("Invalid number choice.")
                    elif y_choice_str in first_df_columns: y_axis_col_chosen = y_choice_str; y_axis_valid = True
                    else: print(f"Column '{y_choice_str}' not found.")
                    if y_axis_valid and y_axis_col_chosen == x_axis_col_chosen: print(f"Note: Plotting '{y_axis_col_chosen}' against itself.")


            if x_axis_valid and y_axis_valid:
                plot_multiple_polar_data(data_to_display, x_axis_col_chosen, y_axis_col_chosen)
            else:
                print("Could not determine valid X or Y axis for plotting.")
        elif not data_to_display:
            print("No data could be successfully read from the provided polar files.")
        else: # data_to_display exists but first_df_columns is None (all files were empty)
            print("All read polar files were empty or unparsable. Cannot determine columns for plotting.")