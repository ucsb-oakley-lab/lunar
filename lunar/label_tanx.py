# lunar/label_tanx.py

import pandas as pd
import numpy as np

def determine_camera(cX):
    """
    Determines which camera is used based on the cX value.

    Parameters:
    - cX (float): The x-coordinate of the contour.

    Returns:
    - str: 'left' if the x-coordinate is on the left camera, 'right' otherwise.
    """
    return 'left' if cX < 2001 else 'right'

def determine_tank(cX, boundaries):
    """
    Determines the tank based on the cX value and tank boundaries.

    Parameters:
    - cX (float): The x-coordinate of the contour.
    - boundaries (list): List of 8 tank boundaries [t1, t2, t3, t4, t5, t6, t7, t8].

    Returns:
    - str: The tank label or 'noise'.
    """
    t1, t2, t3, t4, t5, t6, t7, t8 = boundaries

    if cX < t1:
        return 'noise'
    elif t1 <= cX < t2:
        return 'left_tank1'
    elif t2 <= cX < t3:
        return 'left_tank2'
    elif t3 <= cX < t4:
        return 'left_tank3'
    elif t4 <= cX < t5:
        return 'noise'
    elif t5 <= cX < t6:
        return 'right_tank1'
    elif t6 <= cX < t7:
        return 'right_tank2'
    elif t7 <= cX <= t8:
        return 'right_tank3'
    else:
        return 'noise'

def calculate_cXtank(cX, tank, boundaries):
    """
    Calculates the cXtank value based on the tank and boundaries.

    Parameters:
    - cX (float): The x-coordinate of the contour.
    - tank (str): The determined tank label.
    - boundaries (list): List of 8 tank boundaries [t1, t2, t3, t4, t5, t6, t7, t8].

    Returns:
    - float: The relative x-coordinate within the tank, or NaN if not applicable.
    """
    t1, t2, t3, t4, t5, t6, t7, t8 = boundaries
    if tank == 'left_tank1':
        return cX - t1
    elif tank == 'left_tank2':
        return cX - t2
    elif tank == 'left_tank3':
        return cX - t3
    elif tank == 'right_tank1':
        return cX - t5
    elif tank == 'right_tank2':
        return cX - t6
    elif tank == 'right_tank3':
        return cX - t7
    else:
        return np.nan

def analyze_contours(input_file, tank_boundaries):
    """
    Analyzes the input contour data to label tanks and remove glare.

    Parameters:
    - input_file (str): Path to the input tab-delimited file.
    - tank_boundaries (list): List of 8 tank boundaries [t1, t2, t3, t4, t5, t6, t7, t8].

    Returns:
    - DataFrame: A modified DataFrame with additional columns for camera, tank, and cXtank.
    """
    # Read the input file
    df = pd.read_csv(input_file, delimiter='\t')

    # Remove rows labeled as glare
    if 'glare' in df.columns:
        glare_rows_count = df[df['glare'] == 'yes'].shape[0]
        df = df[df['glare'] != 'yes']
        print(f"Removed {glare_rows_count} rows labeled as glare.")

    # Add the 'camera' column
    df['camera'] = df['cX'].apply(determine_camera)

    # Add the 'tank' column based on the boundaries provided
    df['tank'] = df.apply(lambda row: determine_tank(row['cX'], tank_boundaries), axis=1)

    # Add the 'cXtank' column by subtracting the appropriate tank boundary from cX
    df['cXtank'] = df.apply(lambda row: calculate_cXtank(row['cX'], row['tank'], tank_boundaries), axis=1)

    # Output the modified DataFrame to a new CSV file
    output_file = 'analyzed_' + input_file
    df.to_csv(output_file, sep='\t', index=False)

    print(f"Analysis complete. Results saved to {output_file}")
    return df

