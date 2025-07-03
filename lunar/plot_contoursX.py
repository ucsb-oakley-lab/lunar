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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot frame vs. cX from a contour data file.")
    parser.add_argument("-f", "--file", required=True, help="Path to tab-delimited contour file")
    parser.add_argument("-g", "--glare", action="store_true", help="Color points based on 'glare' column if present")
    parser.add_argument("--zoomx", nargs=2, type=int, metavar=("XMIN", "XMAX"),
                        help="Zoom x-axis range (e.g., --zoomx 1000 2000)")
    parser.add_argument("--zoomy", nargs=2, type=int, metavar=("YMIN", "YMAX"),
                        help="Zoom y-axis range (e.g., --zoomy 300 400)")

    args = parser.parse_args()

    zoomx_tuple = tuple(args.zoomx) if args.zoomx else None
    zoomy_tuple = tuple(args.zoomy) if args.zoomy else None

    plot_contours(
        file_path=args.file,
        glare=args.glare,
        zoomx=zoomx_tuple,
        zoomy=zoomy_tuple
    )

