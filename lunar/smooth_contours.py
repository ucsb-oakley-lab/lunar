# lunar/smooth_contours.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def smooth_contours(input_file, output_file=None, window=10, pad=False):
    """
    Plots the overall average number of contours per frame with clustering for active and inactive periods.

    Parameters:
    - input_file (str): Path to the CSV file.
    - output_file (str, optional): Name of the output plot file (optional).
    - window (int, optional): Window size for smoothing (default: 10 frames).
    - pad (bool, optional): Whether to pad early frames with zeros to avoid edge effects (default: False).
    """
    # Read the CSV file
    df = pd.read_csv(input_file, delimiter='\t')

    # Calculate the number of contours per frame for each tank
    df['contour_count'] = df.groupby(['frame', 'tank'])['tank'].transform('count')

    # Create a DataFrame with the number of contours per frame for each tank
    contour_counts = df.groupby(['frame', 'tank']).size().unstack(fill_value=0)

    # Fill missing frames with zero contours
    contour_counts = contour_counts.reindex(range(contour_counts.index.min(), contour_counts.index.max() + 1), fill_value=0)

    # Calculate the average number of contours for each pair of tanks
    contour_counts['tank1_avg'] = contour_counts[['left_tank1', 'right_tank1']].mean(axis=1)
    contour_counts['tank2_avg'] = contour_counts[['left_tank2', 'right_tank2']].mean(axis=1)
    contour_counts['tank3_avg'] = contour_counts[['left_tank3', 'right_tank3']].mean(axis=1)

    # Calculate the overall average across all three tanks
    contour_counts['overall_avg'] = contour_counts[['tank1_avg', 'tank2_avg', 'tank3_avg']].mean(axis=1)

    # Apply zero padding to handle edge effects
    if pad:
        # Pad the beginning of the data with zeros
        padding = pd.DataFrame(0, index=range(-window + 1, 0), columns=['overall_avg'])
        padded_data = pd.concat([padding, contour_counts[['overall_avg']]])
        # Apply smoothing using a running average
        smoothed_counts = padded_data.rolling(window=window, min_periods=window).mean()
    else:
        # Apply smoothing using a running average without padding
        smoothed_counts = contour_counts[['overall_avg']].rolling(window=window, min_periods=1).mean()

    # Prepare the data for K-means clustering
    combined_data = smoothed_counts['overall_avg'].values.reshape(-1, 1)

    # Remove any NaN values before clustering
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

    # Apply K-means clustering to classify frames into "active" and "inactive"
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(combined_data)
    labels = kmeans.labels_

    # Determine which cluster represents "active" (higher average contour count)
    active_cluster = np.argmax(kmeans.cluster_centers_)

    # Map the labels back to the smoothed data
    active_labels = (labels == active_cluster).astype(int)

    # Create a DataFrame to hold the active labels
    smoothed_counts = smoothed_counts.dropna()  # Drop rows with NaN values after smoothing
    smoothed_counts['active'] = active_labels

    # Convert indices and data to numpy arrays for plotting
    frames = np.array(smoothed_counts.index)
    overall_avg = np.array(smoothed_counts['overall_avg'])

    # Find y-axis limits for the plot
    y_min = np.nanmin(overall_avg)
    y_max = np.nanmax(overall_avg)

    # Plot the overall smoothed average contour count with background shading for active periods
    plt.figure(figsize=(12, 6))
    plt.plot(frames, overall_avg, color='blue', alpha=0.7)

    # Shade the active periods
    for i in range(len(frames)):
        if smoothed_counts['active'].iloc[i] == 1:  # Active frames
            plt.axvspan(frames[i] - 0.5, frames[i] + 0.5, color='lightgray', alpha=0.5)

    # Customize the plot
    plt.title("Overall Average Number of Contours per Frame (Smoothed)")
    plt.xlabel("Frame")
    plt.ylabel("Average Number of Contours")
    plt.ylim(y_min, y_max)
    plt.grid(True)

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

