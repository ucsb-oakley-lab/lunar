import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Overlay plot of contours for two camera views with comparison.')
    parser.add_argument('filename', type=str, help='Path to the tab-separated input file')
    args = parser.parse_args()

    # Read the input file
    df = pd.read_csv(args.filename, sep='\t')

    # Filter data for each tank
    left_tank_data = df[df['tank'] == 'left_tank1']
    right_tank_data = df[df['tank'] == 'right_tank1']

    # Filter data where match_status is 'match'
    match_data = df[df['match_status'] == 'match']
    left_tank_match = match_data[match_data['tank'] == 'left_tank1']
    right_tank_match = match_data[match_data['tank'] == 'right_tank1']

    # Filter data where match_status is 'xdif'
    xdif_data = df[df['match_status'] == 'xdif']
    left_tank_xdif = xdif_data[xdif_data['tank'] == 'left_tank1']
    right_tank_xdif = xdif_data[xdif_data['tank'] == 'right_tank1']

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Plot the original overlay for all data with frame on x-axis and cXtank on y-axis
    axes[0].scatter(left_tank_data['frame'], left_tank_data['cXtank'], label='Left Tank 1', color='blue', alpha=0.5)
    axes[0].scatter(right_tank_data['frame'], right_tank_data['cXtank'], label='Right Tank 1', color='red', alpha=0.5)
    axes[0].set_title('Overlay of All Contours')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('cXtank')
    axes[0].legend()

    # Plot the overlay for data where match_status is 'match'
    axes[1].scatter(left_tank_match['frame'], left_tank_match['cXtank'], label='Left Tank 1 (Match)', color='blue', alpha=0.5)
    axes[1].scatter(right_tank_match['frame'], right_tank_match['cXtank'], label='Right Tank 1 (Match)', color='red', alpha=0.5)
    axes[1].set_title('Overlay of Matching Contours')
    axes[1].set_xlabel('Frame')
    axes[1].legend()

    # Plot the overlay for data where match_status is 'xdif'
    axes[2].scatter(left_tank_xdif['frame'], left_tank_xdif['cXtank'], label='Left Tank 1 (X-Diff)', color='blue', alpha=0.5)
    axes[2].scatter(right_tank_xdif['frame'], right_tank_xdif['cXtank'], label='Right Tank 1 (X-Diff)', color='red', alpha=0.5)
    axes[2].set_title('Overlay of X-Difference Contours')
    axes[2].set_xlabel('Frame')
    axes[2].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

