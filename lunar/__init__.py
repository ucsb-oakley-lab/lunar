from .find_contours import find_contours_from_videos
from .plot_contours import plot_contours
from .identify_glare import normalize_data, cluster_data, process_large_file, clip_ends, manual_mark_glare, concatenate_and_cluster
from .plot_glare_contours import plot_glare_contours
from .label_tanx import determine_camera, determine_tank, calculate_cXtank, analyze_contours
from .match_cameras import match_cameras
from .plot_matched import plot_matched
from .smooth_contours import smooth_contours
from .plot_days import plot_days
from .add_time import add_time  # Import your new function here

__all__ = [
    'find_contours_from_videos', 'plot_contours', 'normalize_data', 'cluster_data',
    'process_large_file', 'clip_ends', 'manual_mark_glare', 'plot_glare_contours', 'determine_camera', 
    'determine_tank', 'calculate_cXtank', 'analyze_contours', 'match_cameras',
    'plot_matched', 'smooth_contours', 'concatenate_and_cluster', 'plot_days', 'add_time'  # Add it to the __all__ list
]

