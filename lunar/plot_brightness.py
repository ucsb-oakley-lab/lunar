import pandas as pd
import matplotlib.pyplot as plt

def plot_contours(file_path, zoomx=None, zoomy=None, dark=False):
    """
    Plots frame vs. brightness from a tab-delimited file.

    Parameters:
    - file_path (str): Path to the tab-delimited file.
    - zoomx (tuple): A tuple specifying the x-axis range (min, max) for a zoomed-in plot.
    - zoomy (tuple): A tuple specifying the y-axis range (min, max) for a zoomed-in plot.
    - dark (bool): If True, plot on black background with white dots and no labels or grid.
    """
    data = pd.read_csv(file_path, delimiter='\t')
    frame = data['frame'].to_numpy()
    bri = data['brightness'].to_numpy()

    # Plot full range
    plt.figure()
    if dark:
        plt.style.use('dark_background')
        plt.scatter(frame, bri, marker='o', s=1, c='white')
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_facecolor('black')
    else:
        plt.scatter(frame, bri, marker='o', s=1, c='black')
        plt.xlabel('Frame')
        plt.ylabel('Brightness')
        plt.title('Frame vs. Brightness')
        plt.grid(True)
    plt.show()

    # Plot zoomed range
    if zoomx or zoomy:
        plt.figure()
        if dark:
            plt.style.use('dark_background')
            plt.scatter(frame, bri, marker='o', s=1, c='white')
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_facecolor('black')
        else:
            plt.scatter(frame, bri, marker='o', s=1, c='black')
            plt.xlabel('Frame')
            plt.ylabel('Brightness')
            zoom_title = 'Zoomed'
            if zoomx:
                zoom_title += f' (X: {zoomx[0]} to {zoomx[1]})'
            if zoomy:
                zoom_title += f' (Y: {zoomy[0]} to {zoomy[1]})'
            plt.title(f'Frame vs. cX {zoom_title}')
            plt.grid(True)
        if zoomx:
            plt.xlim(zoomx)
        if zoomy:
            plt.ylim(zoomy)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot frame vs. brightness from a contour data file.")
    parser.add_argument("-f", "--file", required=True, help="Path to tab-delimited contour file")
    parser.add_argument("--zoomx", nargs=2, type=int, metavar=("XMIN", "XMAX"),
                        help="Zoom x-axis range (e.g., --zoomx 1000 2000)")
    parser.add_argument("--zoomy", nargs=2, type=int, metavar=("YMIN", "YMAX"),
                        help="Zoom y-axis range (e.g., --zoomy 300 400)")
    parser.add_argument("--dark", type=int, default=0, help="Set to 1 for black background with white dots")

    args = parser.parse_args()

    zoomx_tuple = tuple(args.zoomx) if args.zoomx else None
    zoomy_tuple = tuple(args.zoomy) if args.zoomy else None
    dark_mode = bool(args.dark)

    plot_contours(
        file_path=args.file,
        zoomx=zoomx_tuple,
        zoomy=zoomy_tuple,
        dark=dark_mode
    )

