# lunar/plot_contours.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_contours(file_path):
    """
    Plots frame vs. cX from a tab-delimited file containing contour data.

    Parameters:
    - file_path (str): Path to the tab-delimited file.
    """
    # Load the file into a DataFrame
    data = pd.read_csv(file_path, delimiter='\t')

    # Convert columns to numpy arrays before plotting
    frame = data['frame'].to_numpy()
    cX = data['cX'].to_numpy()

    # Plot the data as individual points
    plt.scatter(frame, cX, marker='o', s=5)
    plt.xlabel('Frame')
    plt.ylabel('cX')
    plt.title('Frame vs. cX')
    plt.grid(True)
    plt.show()

