# lunar/identify_glare.py

import pandas as pd
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def normalize_data(data):
    """
    Normalizes cX, cY, frame, and area columns using StandardScaler.

    Parameters:
    - data (DataFrame): The chunk of the input data to normalize.

    Returns:
    - ndarray: Normalized data as a NumPy array.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[['cX', 'cY', 'frame', 'area']])
    return normalized_data

def cluster_data(data, eps, min_samples):
    """
    Applies DBSCAN clustering to the data.

    Parameters:
    - data (ndarray): Normalized data for clustering.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
    - ndarray: Cluster labels for each sample in the data.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)
    return cluster_labels

def concatenate_and_cluster(file_path_pattern: str, n_clusters: int, output_file: str) -> None:
    """
    Concatenate multiple tab-delimited files, perform k-means clustering on the 'average_contours' column,
    and write the result to an output file.
    
    Parameters:
    - file_path_pattern: str, file path pattern to match multiple files using an asterisk (e.g., 'data/*.txt').
    - n_clusters: int, the number of clusters to create with k-means.
    - output_file: str, the filename to save the concatenated and clustered data.
    
    Returns:
    - None
    """
    # Step 1: Concatenate all files matching the pattern
    file_list = glob.glob(file_path_pattern)
    df_list = [pd.read_csv(file, delimiter='\t') for file in file_list]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Step 2: Perform k-means clustering on 'average_contours'
    kmeans = KMeans(n_clusters=n_clusters, n_init = 'auto', random_state=42)
    combined_df['kclusters'] = kmeans.fit_predict(combined_df[['average_contours']])
    
    # Step 3: Write the result to the output file
    combined_df.to_csv(output_file, sep='\t', index=False)

# Example usage:
# concatenate_and_cluster('data/*.txt', 3, 'clustered_output.txt')


def process_large_file(input_file, output_file, min_cluster_size, eps=0.5, min_samples=5, chunksize=100000):
    """
    Processes a large input file in chunks, applying DBSCAN clustering to identify glare.

    Parameters:
    - input_file (str): Path to the input tab-delimited file.
    - output_file (str): Path to the output file.
    - min_cluster_size (int): Minimum number of elements in a cluster to be considered glare.
    - eps (float, optional): Epsilon parameter for DBSCAN (default: 0.5).
    - min_samples (int, optional): Minimum number of samples for a cluster in DBSCAN (default: 5).
    - chunksize (int, optional): Chunk size for processing large files (default: 100000).
    """
    # Prepare to write the output file
    with pd.read_csv(input_file, sep='\t', chunksize=chunksize, 
                     dtype={'cX': np.float32, 'cY': np.float32, 'frame': np.int32, 'area': np.float32}) as reader, \
         open(output_file, 'w') as output:
        first_chunk = True

        for chunk in reader:
            # Normalize the data for clustering
            normalized_data = normalize_data(chunk)

            # Cluster the data
            cluster_labels = cluster_data(normalized_data, eps=eps, min_samples=min_samples)

            # Add cluster labels to the original dataframe
            chunk['cluster'] = cluster_labels

            # Determine the size of each cluster
            cluster_counts = chunk['cluster'].value_counts()

            # Add the 'glare' column based on the cluster size
            chunk['glare'] = chunk['cluster'].apply(
                lambda x: 'yes' if (x != -1 and cluster_counts[x] >= min_cluster_size) else 'no'
            )

            # Remove the 'cluster' column for the output
            chunk.drop(columns=['cluster'], inplace=True)

            # Write the processed chunk to the output file
            chunk.to_csv(output, sep='\t', index=False, header=first_chunk, mode='a')
            first_chunk = False

import pandas as pd

def manual_mark_glare(input_file, output_file, low_clip, hi_clip, hmark=None):
    # Read the input file into a DataFrame
    df = pd.read_csv(input_file, sep='\t')
    
    # Mark the 'glare' column as 'yes' where 'frame' is less than low_clip or more than hi_clip
    df.loc[(df['frame'] < low_clip) | (df['frame'] > hi_clip), 'glare'] = 'yes'
    
    # Process hmark if provided
    if hmark is not None:
        # If hmark is a string, split it into numbers
        if isinstance(hmark, str):
            # Remove any spaces and split by commas
            hmark_list = hmark.replace(' ', '').split(',')
            # Convert to floats
            hmark_numbers = [float(x) for x in hmark_list]
        elif isinstance(hmark, (list, tuple)):
            hmark_numbers = hmark
        else:
            raise ValueError("hmark must be a string or a list/tuple of numbers")
        
        # Ensure hmark has pairs of low_mark and hi_mark
        if len(hmark_numbers) % 2 != 0:
            raise ValueError("hmark must contain pairs of low_mark and hi_mark values")
        
        # Iterate over pairs and mark 'glare' accordingly
        for i in range(0, len(hmark_numbers), 2):
            mark1 = hmark_numbers[i]
            mark2 = hmark_numbers[i+1]
            low_mark = min(mark1, mark2)
            hi_mark = max(mark1, mark2)
            # Mark 'glare' as 'yes' where cX is between low_mark and hi_mark
            df.loc[(df['cX'] > low_mark) & (df['cX'] < hi_mark), 'glare'] = 'yes'
    
    # Write the modified DataFrame to the output file
    df.to_csv(output_file, sep='\t', index=False)

def clip_ends(input_file, output_file, low_clip, hi_clip):
    # Read the input file into a DataFrame
    df = pd.read_csv(input_file, sep='\t')
    df.head()
    # Mark the 'glare' column as 'yes' where 'frame' is less than low_clip or more than hi_clip
    df.loc[(df['frame'] < low_clip) | (df['frame'] > hi_clip), 'glare'] = 'yes'

    # Write the modified DataFrame to the output file
    df.to_csv(output_file, sep='\t', index=False)


def check_vertical_glare(data, vertical_glare_threshold, frame_range, cy_threshold_count, cy_cutoff, low_clip=None, hi_clip=None):
    """
    Checks for the 'glare' column in the data, adds it if missing, and marks entire ranges of frames
    as 'yes' for glare if they meet the criteria:
    - A minimum total number of contours in the range.
    - A minimum number of contours in the range with cY > cy_cutoff.

    Also, sets glare to 'yes' for rows outside the specified frame range using low_clip and hi_clip.

    Parameters:
    - data (DataFrame): The input data to check and modify.
    - vertical_glare_threshold (int): The minimum number of total contours required in a range to consider it for glare.
    - frame_range (int): The range (window size) of frames to sum up for sliding window analysis.
    - cy_threshold_count (int): The minimum number of contours with cY exceeding the cutoff required to mark 
      the range of frames as 'glare'.
    - cy_cutoff (float): The cutoff value for cY to consider it for glare.
    - low_clip (int, optional): The lower frame limit; rows with frame < low_clip will have glare set to 'yes'.
    - hi_clip (int, optional): The upper frame limit; rows with frame > hi_clip will have glare set to 'yes'.

    Returns:
    - DataFrame: The modified data with the 'glare' column updated.
    """
    # Check if the 'glare' column exists, add it with default 'no' values if missing
    if 'glare' not in data.columns:
        data['glare'] = 'no'

    # Set glare to 'yes' for rows outside the specified frame range if not already 'yes'
    if low_clip is not None:
        data.loc[(data['frame'] < low_clip) & (data['glare'] != 'yes'), 'glare'] = 'yes'
    if hi_clip is not None:
        data.loc[(data['frame'] > hi_clip) & (data['glare'] != 'yes'), 'glare'] = 'yes'

    # Sort the data by frame number
    data.sort_values(by='frame', inplace=True)

    # Determine the maximum frame and the number of padding rows needed
    max_frame = data['frame'].max()
    padding_needed = frame_range - (max_frame % frame_range)

    # Create dummy padding data
    dummy_data = pd.DataFrame({
        'frame': range(max_frame + 1, max_frame + 1 + padding_needed),
        'cY': [0] * padding_needed,  # Assuming 0 will not affect cY cutoff
        'glare': ['no'] * padding_needed
    })

    # Append dummy data to the original data
    data = pd.concat([data, dummy_data], ignore_index=True)

    # Create a rolling count of contours within the frame range
    data['contours_count'] = data.groupby('frame')['frame'].transform('size')

    # Create a rolling count of contours with cY > cy_cutoff within the frame range
    data['contours_above_cutoff_count'] = data.groupby('frame')['cY'].transform(lambda x: (x > cy_cutoff).sum())

    # Calculate a rolling sum over the specified frame range
    total_contours_rolling = data['contours_count'].rolling(window=frame_range, min_periods=1).sum()
    contours_above_cutoff_rolling = data['contours_above_cutoff_count'].rolling(window=frame_range, min_periods=1).sum()

    # Determine which ranges should be marked as 'glare'
    glare_mask = (total_contours_rolling >= vertical_glare_threshold) & (contours_above_cutoff_rolling >= cy_threshold_count)

    # Ensure that only rows not already marked as 'yes' are updated
    data.loc[glare_mask & (data['glare'] != 'yes'), 'glare'] = 'yes'

    # Remove the dummy data
    data = data[data['frame'] <= max_frame].copy()

    # Clean up temporary columns
    data.drop(columns=['contours_count', 'contours_above_cutoff_count'], inplace=True)

    # Ensure that rows previously marked as 'yes' are not overwritten
    data['glare'] = data.groupby('frame')['glare'].transform(lambda x: 'yes' if 'yes' in x.values else 'no')

    return data

def check_vertical_glareOLD(data, vertical_glare_threshold, frame_range):
    """
    Checks for the 'glare' column in the data, adds it if missing, 
    and marks frames with occurrences exceeding the threshold in a sliding window range 
    as 'yes' for glare.

    Parameters:
    - data (DataFrame): The input data to check and modify.
    - vertical_glare_threshold (int): The threshold value for frame occurrences.
    - frame_range (int): The range (window size) of frames to sum up for sliding window analysis.

    Returns:
    - DataFrame: The modified data with the 'glare' column updated.
    """
    # Check if the 'glare' column exists, add it with default 'no' values if missing
    if 'glare' not in data.columns:
        data['glare'] = 'no'

    # Sort the data by frame number
    data.sort_values(by='frame', inplace=True)

    # Count occurrences of each frame
    frame_counts = data['frame'].value_counts().sort_index()

    # Compute a rolling sum over the specified frame range
    rolling_sum = frame_counts.rolling(window=frame_range, min_periods=1).sum()

    # Find frames where the rolling sum exceeds the threshold
    glare_frames = rolling_sum[rolling_sum > vertical_glare_threshold].index

    # Mark 'glare' as 'yes' for rows where the frame is in glare_frames
    data.loc[data['frame'].isin(glare_frames), 'glare'] = 'yes'

    return data
def vertical_glare_fileio(input_file, output_file, vertical_glare_threshold, frame_range, cy_threshold_count, cy_cutoff, low_clip, hi_clip):
    """
    Reads a tab-delimited file, checks for the 'glare' column, adds it if missing, 
    and marks frames with occurrences exceeding the threshold in a sliding window range 
    as 'yes' for glare. Saves the updated data to an output file.

    Parameters:
    - input_file (str): Path to the input tab-delimited file.
    - output_file (str): Path to the output tab-delimited file.
    - vertical_glare_threshold (int): The threshold value for frame occurrences.
    - frame_range (int): The range (window size) of frames to sum up for sliding window analysis.
    """
    # Read the input file
    data = pd.read_csv(input_file, sep='\t', dtype={'cX': np.float32, 'cY': np.float32, 'frame': np.int32, 'area': np.float32})

    # Check and mark glare using the modified sliding window function
    data = check_vertical_glare(data, vertical_glare_threshold, frame_range, cy_threshold_count, cy_cutoff, low_clip, hi_clip)

    # Save the updated data to the output file
    data.to_csv(output_file, sep='\t', index=False)

