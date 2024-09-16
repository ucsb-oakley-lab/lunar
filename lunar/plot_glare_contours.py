# lunar/plot_glare_contours.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_glare_contours(input_file, color_by_cluster=False):
    """
    Scatter plot of Frame vs. cX, colored by glare or cluster.

    Parameters:
    - input_file (str): Path to the input tab-delimited file.
    - color_by_cluster (bool): If True, color by cluster; otherwise, color by glare.
    
    Returns:
    - None
    """
    # Load the input file
    df = pd.read_csv(input_file, sep='\t')

    plt.figure(figsize=(10, 6))

    if color_by_cluster:
        # Color by cluster, distinguishing cluster '-1' from others
        cluster_negative_one = df[df['cluster'] == -1]
        cluster_other = df[df['cluster'] != -1]

        plt.scatter(
            cluster_negative_one['frame'],
            cluster_negative_one['cX'],
            color='red',
            s=1,  # Reduced point size
            label="Cluster -1"
        )

        plt.scatter(
            cluster_other['frame'],
            cluster_other['cX'],
            color='blue',
            s=1,  # Reduced point size
            label="Other Clusters"
        )

        plt.legend(loc='upper right', title="Clusters", fontsize='small', markerscale=1.2)

    else:
        # Color by glare
        df['glare_color'] = df['glare'].apply(lambda x: 'red' if x == 'yes' else 'blue')

        # Plot the data based on glare
        for glare_value in df['glare'].unique():
            glare_data = df[df['glare'] == glare_value]
            plt.scatter(
                glare_data['frame'],
                glare_data['cX'],
                color='red' if glare_value == 'yes' else 'blue',
                s=1,  # Reduced point size
                label=f"Glare: {glare_value}"
            )
        plt.legend(loc='upper right', title="Glare", fontsize='small', markerscale=1.2)

    plt.xlabel('Frame')
    plt.ylabel('cX')
    plt.title('Scatter Plot of Frame vs. cX')
    plt.show()

