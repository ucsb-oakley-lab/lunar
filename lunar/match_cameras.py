# lunar/match_cameras.py

import pandas as pd
import numpy as np

def match_cameras(input_file, output_file, distance_x=200, distance_y=200):
    """
    Matches camera data based on cX and cY values and writes the updated DataFrame to an output file.

    Parameters:
    - input_file (str): Path to the input tab-delimited file.
    - output_file (str): Path to the output file.
    - distance_x (float): Maximum allowed difference for cX values.
    - distance_y (float): Maximum allowed difference for cY values.
    """
    # Read the CSV file
    df = pd.read_csv(input_file, delimiter='\t')

    # Filter out rows where 'tank' is 'noise'
    df = df[df['tank'] != 'noise']

    # Add a new column for the match status
    df['match_status'] = ""

    # Function to find the closest pairs between two lists considering both cX and cY values
    def find_closest_pairs(left, right, left_indices, right_indices):
        pairs = []
        used_right_indices = set()
        for left_val, left_idx in zip(left, left_indices):
            closest_idx = None
            closest_dist = float('inf')
            for idx, (right_val, right_idx) in enumerate(zip(right, right_indices)):
                if idx in used_right_indices:
                    continue
                # Calculate Euclidean distance between the left and right points
                dist = np.sqrt((left_val[0] - right_val[0])**2 + (left_val[1] - right_val[1])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx
            if closest_idx is not None:
                pairs.append((left_val, right[closest_idx], left_idx, right_indices[closest_idx]))
                used_right_indices.add(closest_idx)
        return pairs

    # Prepare lists to store matched, included, and excluded pairs for each tank
    included_pairs = {f'tank{i}': [] for i in range(1, 4)}
    excluded_pairs_x = {f'tank{i}': [] for i in range(1, 4)}
    excluded_pairs_y = {f'tank{i}': [] for i in range(1, 4)}
    excluded_pairs_both = {f'tank{i}': [] for i in range(1, 4)}

    # Group by frame and process each group
    for frame, group in df.groupby('frame'):
        for tank_num in range(1, 4):
            left_tank = group[group['tank'] == f'left_tank{tank_num}'][['cXtank', 'cY']].values
            right_tank = group[group['tank'] == f'right_tank{tank_num}'][['cXtank', 'cY']].values
            left_indices = group[group['tank'] == f'left_tank{tank_num}'].index
            right_indices = group[group['tank'] == f'right_tank{tank_num}'].index

            pairs = find_closest_pairs(left_tank, right_tank, left_indices, right_indices)

            for (left_val, right_val, left_idx, right_idx) in pairs:
                left_val_x, left_val_y = left_val
                right_val_x, right_val_y = right_val
                x_diff_too_high = abs(left_val_x - right_val_x) > distance_x
                y_diff_too_high = abs(left_val_y - right_val_y) > distance_y

                if x_diff_too_high and y_diff_too_high:
                    excluded_pairs_both[f'tank{tank_num}'].append((left_val, right_val))
                    df.at[left_idx, 'match_status'] = 'bothdif'
                    df.at[right_idx, 'match_status'] = 'bothdif'
                elif x_diff_too_high:
                    excluded_pairs_x[f'tank{tank_num}'].append((left_val, right_val))
                    df.at[left_idx, 'match_status'] = 'xdif'
                    df.at[right_idx, 'match_status'] = 'xdif'
                elif y_diff_too_high:
                    excluded_pairs_y[f'tank{tank_num}'].append((left_val, right_val))
                    df.at[left_idx, 'match_status'] = 'ydif'
                    df.at[right_idx, 'match_status'] = 'ydif'
                else:
                    included_pairs[f'tank{tank_num}'].append((left_val, right_val))
                    df.at[left_idx, 'match_status'] = 'match'
                    df.at[right_idx, 'match_status'] = 'match'

    # Write the updated DataFrame to a new tab-delimited file
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Updated data has been written to {output_file}")

