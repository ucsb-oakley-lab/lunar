# barplot_contours.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_contour_histogram(file_path, zoomx=None, zoomy=None, dark=False):
    """
    Plots a bar chart of the number of contours per frame.

    Parameters:
    - file_path (str): Path to tab-delimited contour data file.
    - zoomx (tuple): Optional x-axis zoom range (xmin, xmax).
    - zoomy (tuple): Optional y-axis zoom range (ymin, ymax).
    - dark (bool): If True, use dark mode for plotting.
    """
    df = pd.read_csv(file_path, delimiter="\t")
    df.columns = df.columns.str.strip().str.lstrip("#").str.strip()

    if 'frame' not in df.columns:
        raise ValueError(f"Available columns: {df.columns.tolist()}\nMissing required 'frame' column.")

    frame_counts = df['frame'].value_counts().sort_index()

    # Start plotting
    # Determine figure width: 1 pixel per frame at dpi=100, capped for sanity
    n_frames = frame_counts.index.max() - frame_counts.index.min() + 1
    width_inches = min(n_frames / 100, 200)  # 100 frames per inch; cap at 200 inches wide

    fig, ax = plt.subplots(figsize=(width_inches, 6), dpi=100)



    if dark:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        bar_color = 'white'
        bar_width = 1
    else:
        bar_color = 'black'
        bar_width = 1

    ax.bar(frame_counts.index, frame_counts.values, width=bar_width, color=bar_color)

    if not dark:
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number of Contours')
        ax.set_title('Contours per Frame')
        ax.grid(True, axis='y')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if zoomx:
        ax.set_xlim(zoomx)
    if zoomy:
        ax.set_ylim(zoomy)

    plt.tight_layout()
    plt.show()

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot bar chart of number of contours per frame.")
    parser.add_argument("-f", "--file", required=True, help="Path to tab-delimited contour file")
    parser.add_argument("--zoomx", nargs=2, type=int, metavar=("XMIN", "XMAX"),
                        help="Zoom x-axis range (e.g., --zoomx 1000 2000)")
    parser.add_argument("--zoomy", nargs=2, type=int, metavar=("YMIN", "YMAX"),
                        help="Zoom y-axis range (e.g., --zoomy 0 50)")
    parser.add_argument("--dark", action="store_true", help="Enable dark mode for black background and white bars")

    args = parser.parse_args()
    zoomx_tuple = tuple(args.zoomx) if args.zoomx else None
    zoomy_tuple = tuple(args.zoomy) if args.zoomy else None

    plot_contour_histogram(
        file_path=args.file,
        zoomx=zoomx_tuple,
        zoomy=zoomy_tuple,
        dark=args.dark
    )

