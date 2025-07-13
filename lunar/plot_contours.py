import pandas as pd
import matplotlib.pyplot as plt

def plot_contours(file_path, match=False, zoomx=None, zoomy=None):
    """
    Plots frame vs. cX from a tab-delimited file containing contour data.

    Parameters:
    - file_path (str): Path to the tab-delimited file.
    - match (bool): If True, color points based on the 'match' column values.
    - zoomx (tuple): A tuple specifying the x-axis range (min, max) for a zoomed-in plot.
    - zoomy (tuple): A tuple specifying the y-axis range (min, max) for a zoomed-in plot.
    """
    data = pd.read_csv(file_path, delimiter='\t')
    frame = data['frame'].to_numpy()
    cX = data['cX'].to_numpy()

    # Determine colors
    if match and 'match' in data.columns:
        color_map = {'N': 'blue', 'L': 'lightgray', 'R': 'dimgray'}
        match_colors = data['match'].map(color_map).fillna('black').to_numpy()
        color_key = match_colors
    else:
        color_key = 'black'

    # Plot full range
    plt.figure()
    plt.scatter(frame, cX, marker='o', s=1, c=color_key)
    plt.xlabel('Frame')
    plt.ylabel('cX')
    plt.title('Frame vs. cX (match colored)' if match else 'Frame vs. cX')
    plt.grid(True)
    plt.show()

    # Plot zoomed range
    if zoomx or zoomy:
        plt.figure()
        plt.scatter(frame, cX, marker='o', s=1, c=color_key)
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
    parser.add_argument("-m", "--match", action="store_true", help="Color points based on 'match' column")
    parser.add_argument("--zoomx", nargs=2, type=int, metavar=("XMIN", "XMAX"),
                        help="Zoom x-axis range (e.g., --zoomx 1000 2000)")
    parser.add_argument("--zoomy", nargs=2, type=int, metavar=("YMIN", "YMAX"),
                        help="Zoom y-axis range (e.g., --zoomy 300 400)")

    args = parser.parse_args()

    zoomx_tuple = tuple(args.zoomx) if args.zoomx else None
    zoomy_tuple = tuple(args.zoomy) if args.zoomy else None

    plot_contours(
        file_path=args.file,
        match=args.match,
        zoomx=zoomx_tuple,
        zoomy=zoomy_tuple
    )

