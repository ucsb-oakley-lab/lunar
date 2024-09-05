# lunar/plot_matched.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_matched(input_file):
    """
    Plots the correlations between left and right tanks based on the matched results.

    Parameters:
    - input_file (str): Path to the tab-delimited input file containing match statuses.
    """
    # Read the input file
    df = pd.read_csv(input_file, delimiter='\t')

    # Prepare lists for matched, included, and excluded pairs
    included_pairs = {f'tank{i}': [] for i in range(1, 4)}
    excluded_pairs_x = {f'tank{i}': [] for i in range(1, 4)}
    excluded_pairs_y = {f'tank{i}': [] for i in range(1, 4)}
    excluded_pairs_both = {f'tank{i}': [] for i in range(1, 4)}

    # Fill lists based on the match status in the DataFrame
    for tank_num in range(1, 4):
        df_left = df[df['tank'] == f'left_tank{tank_num}']
        df_right = df[df['tank'] == f'right_tank{tank_num}']
        
        # Iterate over each row in the left tank data
        for _, left_row in df_left.iterrows():
            # Find matching rows in the right tank data by frame and match status
            matched_right = df_right[(df_right['frame'] == left_row['frame']) & 
                                     (df_right['match_status'] == left_row['match_status'])]

            # Add pairs to the appropriate lists based on the match status
            for _, right_row in matched_right.iterrows():
                pair = ((left_row['cXtank'], left_row['cY']), (right_row['cXtank'], right_row['cY']))
                
                if left_row['match_status'] == 'match':
                    included_pairs[f'tank{tank_num}'].append(pair)
                elif left_row['match_status'] == 'xdif':
                    excluded_pairs_x[f'tank{tank_num}'].append(pair)
                elif left_row['match_status'] == 'ydif':
                    excluded_pairs_y[f'tank{tank_num}'].append(pair)
                elif left_row['match_status'] == 'bothdif':
                    excluded_pairs_both[f'tank{tank_num}'].append(pair)

    # Plotting
    plt.figure(figsize=(15, 10))

    for tank_num in range(1, 4):
        plt.subplot(3, 1, tank_num)
        
        # Plot included pairs
        if included_pairs[f'tank{tank_num}']:
            left_vals, right_vals = zip(*included_pairs[f'tank{tank_num}'])
            left_vals_x, left_vals_y = zip(*left_vals)
            right_vals_x, right_vals_y = zip(*right_vals)
            plt.scatter(left_vals_x, right_vals_x, color='#1f77b4', s=20, alpha=0.7, label='Included')

        # Plot excluded pairs by x difference
        if excluded_pairs_x[f'tank{tank_num}']:
            left_vals_excl, right_vals_excl = zip(*excluded_pairs_x[f'tank{tank_num}'])
            left_vals_excl_x, left_vals_excl_y = zip(*left_vals_excl)
            right_vals_excl_x, right_vals_excl_y = zip(*right_vals_excl)
            plt.scatter(left_vals_excl_x, right_vals_excl_x, color='red', s=20, alpha=0.7, label='Excluded (x)')

        # Plot excluded pairs by y difference
        if excluded_pairs_y[f'tank{tank_num}']:
            left_vals_excl, right_vals_excl = zip(*excluded_pairs_y[f'tank{tank_num}'])
            left_vals_excl_x, left_vals_excl_y = zip(*left_vals_excl)
            right_vals_excl_x, right_vals_excl_y = zip(*right_vals_excl)
            plt.scatter(left_vals_excl_x, right_vals_excl_x, color='yellow', s=20, alpha=0.7, label='Excluded (y)')

        # Plot excluded pairs by both differences
        if excluded_pairs_both[f'tank{tank_num}']:
            left_vals_excl, right_vals_excl = zip(*excluded_pairs_both[f'tank{tank_num}'])
            left_vals_excl_x, left_vals_excl_y = zip(*left_vals_excl)
            right_vals_excl_x, right_vals_excl_y = zip(*right_vals_excl)
            plt.scatter(left_vals_excl_x, right_vals_excl_x, color='orange', s=20, alpha=0.7, label='Excluded (both)')

        plt.xlim(0, 600)
        plt.ylim(0, 600)
        plt.title(f"Correlation between Left Tank {tank_num} and Right Tank {tank_num}")
        plt.xlabel(f"Left Tank {tank_num} cX")
        plt.ylabel(f"Right Tank {tank_num} cX")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

