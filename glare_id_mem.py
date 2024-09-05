#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import argparse

def normalize_data(data):
    # Normalize cX, cY, frame, and area using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[['cX', 'cY', 'frame', 'area']])
    return normalized_data

def cluster_data(data, eps, min_samples):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)
    return cluster_labels

def process_file(input_file, output_file, min_cluster_size, eps=0.5, min_samples=5, chunksize=100000):
    # Prepare to write the output file
    with pd.read_csv(input_file, sep='\t', chunksize=chunksize, dtype={'cX': np.float32, 'cY': np.float32, 'frame': np.int32, 'area': np.float32}) as reader, open(output_file, 'w') as output:
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

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Cluster contour data and label glare.')
    parser.add_argument('input_file', type=str, help='Path to the input tab-delimited file.')
    parser.add_argument('output_file', type=str, help='Path to the output file.')
    parser.add_argument('--min_cluster_size', type=int, default=10000, help='Minimum number of elements in a cluster to be considered glare.')
    parser.add_argument('--eps', type=float, default=0.3, help='Epsilon parameter for DBSCAN.')
    parser.add_argument('-m', '--min_samples', type=int, default=50, help='Minimum number of samples for a cluster in DBSCAN.')
    parser.add_argument('--chunksize', type=int, default=100000, help='Chunk size for processing large files.')
    args = parser.parse_args()

    # Process the file
    process_file(args.input_file, args.output_file, args.min_cluster_size, eps=args.eps, min_samples=args.min_samples, chunksize=args.chunksize)

