import pandas as pd
import matplotlib.pyplot as plt

def plot_contours(file_path, glare=False, zoomx=None, zoomy=None):
    """
    Plots frame vs. cX from a tab-delimited file containing contour data.

    Parameters:
    - file_path (str): Path to the tab-delimited file.
    - glare (bool): If True, color points based on the 'glare' column values.
    - zoomx (tuple): A tuple specifying the x-axis range (min, max) for a zoomed-in plot.
    - zoomy (tuple): A tuple specifying the y-axis range (min, max) for a zoomed-in plot.
    """
    # Load the file into a DataFrame
    data = pd.read_csv(file_path, delimiter='\t')

    # Convert columns to numpy arrays before plotting
    frame = data['frame'].to_numpy()
    cX = data['cX'].to_numpy()

    # Check if glare column should be used
    if glare and 'glare' in data.columns:
        glare_colors = data['glare'].apply(lambda x: 'red' if x == 'yes' else 'blue').to_numpy()
        plt.scatter(frame, cX, marker='o', s=1, c=glare_colors)
    else:
        plt.scatter(frame, cX, marker='o', s=1)

    plt.xlabel('Frame')
    plt.ylabel('cX')
    plt.title('Frame vs. cX')
    plt.grid(True)
    plt.show()

    # If zoomx or zoomy parameters are set, create a zoomed-in plot
    if zoomx or zoomy:
        plt.figure()
        if glare and 'glare' in data.columns:
            plt.scatter(frame, cX, marker='o', s=1, c=glare_colors)
        else:
            plt.scatter(frame, cX, marker='o', s=1)

        # Apply zoom settings
        if zoomx:
            plt.xlim(zoomx)
        if zoomy:
            plt.ylim(zoomy)

        plt.xlabel('Frame')
        plt.ylabel('cX')
        zoom_title = 'Zoomed'
        if zoomx:
            zoom_title += f' (X: {zoomx[0]} to {zoomx[1]})'
        if zoomy:
            zoom_title += f' (Y: {zoomy[0]} to {zoomy[1]})'
        plt.title(f'Frame vs. cX {zoom_title}')
        plt.grid(True)
        plt.show()

