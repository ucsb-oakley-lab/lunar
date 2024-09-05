# lunar/identify_glare.py

import pandas as pd
import numpy as np
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

