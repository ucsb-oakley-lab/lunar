#lunar/smooth_contours.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def smooth_contours(input_file, outfile_suffix=None, window=10, pad=False, date=None):
    """
    Plots the overall average number of contours per frame with clustering for active and inactive periods
    and saves the smoothed data to an output file.

    Parameters:
    - input_file (str): Path to the CSV file.
    - outfile_suffix (str, optional): Suffix for the output files (usually ending in .tsv).
    - window (int, optional): Window size for smoothing (default: 10 frames).
    - pad (bool, optional): Whether to pad early frames with zeros to avoid edge effects (default: False).
    - date (str, optional): Date to be added as a column in the output file.
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
    smoothed_counts = smoothed_counts.dropna()  # Drop rows with NaN values after smoothing
    smoothed_counts['cluster'] = labels
    smoothed_counts['active'] = (labels == active_cluster).astype(int)

    # Create an output DataFrame with the required columns
    output_df = pd.DataFrame({
        'frame': smoothed_counts.index + 1,  # Frame numbers starting at 1
        'average_contours': smoothed_counts['overall_avg'],
        'cluster': smoothed_counts['cluster'],  # KMeans cluster label
        'date': date
    })

    # Determine the output file names based on outfile_suffix
    if outfile_suffix and outfile_suffix.endswith('.tsv'):
        output_file_name = f"smooth_{outfile_suffix}"
        plot_file_name = f"{outfile_suffix[:-4]}.png"
    else:
        output_file_name = f"smooth_{outfile_suffix}.tsv" if outfile_suffix else "smooth_output.tsv"
        plot_file_name = f"{outfile_suffix}.png" if outfile_suffix else "output.png"

    # Save the smoothed data to a TSV file
    output_df.to_csv(output_file_name, index=False, sep='\t')
    print(f"Smoothed data saved to {output_file_name}")

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
    plt.savefig(plot_file_name, bbox_inches='tight')
    print(f"Plot saved to {plot_file_name}")

def smooth_contours_sem(input_file, outfile_suffix=None, window=10, pad=False, date=None):
    """
    Plots the overall average number of contours per frame with clustering for active and inactive periods
    and saves the smoothed data to an output file.

    Parameters:
    - input_file (str): Path to the CSV file.
    - outfile_suffix (str, optional): Suffix for the output files (usually ending in .tsv).
    - window (int, optional): Window size for smoothing (default: 10 frames).
    - pad (bool, optional): Whether to pad early frames with zeros to avoid edge effects (default: False).
    - date (str, optional): Date to be added as a column in the output file.
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

    # **Calculate the Standard Deviation (SD) and Standard Error of the Mean (SEM)**
    contour_counts['std_dev'] = contour_counts[['tank1_avg', 'tank2_avg', 'tank3_avg']].std(axis=1)
    contour_counts['sem'] = contour_counts['std_dev'] / np.sqrt(3)  # Since n=3 tanks

    # Apply zero padding to handle edge effects
    if pad:
        # Pad the beginning of the data with zeros for both mean and SEM
        padding_avg = pd.DataFrame(0, index=range(-window + 1, 0), columns=['overall_avg'])
        padding_sem = pd.DataFrame(0, index=range(-window + 1, 0), columns=['sem'])
        padded_avg = pd.concat([padding_avg, contour_counts[['overall_avg']]])
        padded_sem = pd.concat([padding_sem, contour_counts[['sem']]])
        # Apply smoothing using a running average
        smoothed_avg = padded_avg.rolling(window=window, min_periods=window).mean()
        smoothed_sem = padded_sem.rolling(window=window, min_periods=window).mean()
    else:
        # Apply smoothing using a running average without padding
        smoothed_avg = contour_counts[['overall_avg']].rolling(window=window, min_periods=1).mean()
        smoothed_sem = contour_counts[['sem']].rolling(window=window, min_periods=1).mean()

    # Combine smoothed mean and SEM into one DataFrame
    smoothed_counts = pd.concat([smoothed_avg, smoothed_sem], axis=1)

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
    smoothed_counts = smoothed_counts.dropna()  # Drop rows with NaN values after smoothing
    smoothed_counts['cluster'] = labels
    smoothed_counts['active'] = (labels == active_cluster).astype(int)

    # Create an output DataFrame with the required columns
    output_df = pd.DataFrame({
        'frame': smoothed_counts.index + 1,  # Frame numbers starting at 1
        'average_contours': smoothed_counts['overall_avg'],
        'sem': smoothed_counts['sem'],
        'cluster': smoothed_counts['cluster'],  # KMeans cluster label
        'date': date
    })

    # Determine the output file names based on outfile_suffix
    if outfile_suffix and outfile_suffix.endswith('.tsv'):
        output_file_name = f"smooth_{outfile_suffix}"
        plot_file_name = f"{outfile_suffix[:-4]}.png"
    else:
        output_file_name = f"smooth_{outfile_suffix}.tsv" if outfile_suffix else "smooth_output.tsv"
        plot_file_name = f"{outfile_suffix}.png" if outfile_suffix else "output.png"

    # Save the smoothed data to a TSV file
    output_df.to_csv(output_file_name, index=False, sep='\t')
    print(f"Smoothed data saved to {output_file_name}")

    # Convert indices and data to numpy arrays for plotting
    frames = np.array(smoothed_counts.index)
    overall_avg = np.array(smoothed_counts['overall_avg'])
    sem = np.array(smoothed_counts['sem'])

    # Ensure that SEM is non-negative and handle potential negative values
    sem = np.clip(sem, a_min=0, a_max=None)

    # Calculate the upper and lower bounds for the shaded error region
    lower_bound = overall_avg - sem
    upper_bound = overall_avg + sem

    # Ensure lower_bound is not negative (if your data cannot be negative)
    lower_bound = np.clip(lower_bound, a_min=0, a_max=None)

    # Find y-axis limits for the plot
    y_min = np.nanmin(lower_bound)
    y_max = np.nanmax(upper_bound)

    # Plot the overall smoothed average contour count with shaded SEM region
    plt.figure(figsize=(12, 6))

    # **Plot the gray shading first (active periods)**
    for i in range(len(frames)):
        if smoothed_counts['active'].iloc[i] == 1:  # Active frames
            plt.axvspan(frames[i] - 0.5, frames[i] + 0.5,
                        color='lightgray', alpha=0.5, zorder=1)  # Lower zorder

    # **Plot the shaded SEM region next**
    plt.fill_between(frames, lower_bound, upper_bound,
                     color='blue', alpha=0.2, label='SEM', zorder=2)  # Medium zorder

    # **Plot the mean line on top**
    plt.plot(frames, overall_avg, color='blue', alpha=0.7,
             label='Mean', zorder=3)  # Higher zorder

    # Customize the plot
    plt.title("Overall Average Number of Contours per Frame (Smoothed)")
    plt.xlabel("Frame")
    plt.ylabel("Average Number of Contours")
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.legend()

    # Save or show the plot
    plt.savefig(plot_file_name, bbox_inches='tight')
    print(f"Plot saved to {plot_file_name}")

