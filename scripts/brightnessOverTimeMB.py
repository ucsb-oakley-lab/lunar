#!/usr/bin/env python3

import numpy as np
import cv2
import argparse
import time
import os
from tqdm import tqdm

# Construct the argument parser and parse the arguments
import psutil
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False, default='.',
                help="path to video frame files; default is ./")
ap.add_argument("-o", "--outfile", required=False, default=None,
                help="output filename for brightness results (if not set, auto-generated)")
ap.add_argument("-f", "--file", required=False, default=None,
                help="exact video file name to process (in path)")
ap.add_argument("-g", "--gamma", required=False, default=0.8, type=float,
                help="value of gamma - a correction for human visualization")
ap.add_argument("-b", "--black", required=False, default=0, type=int,
                help="threshold below which is black. Value ranges from 0-255.")
ap.add_argument("-mb", "--multipleb", required=False, nargs=3, type=int,
                help="three black threshold values for multiple adjustments. Values range from 0-255.")
ap.add_argument("-w", "--white", required=False, default=255, type=int,
                help="threshold above which is white. Values range from 0-255")
ap.add_argument("-i", "--initial", required=False, default=1, type=int,
                help="frame to start at; default = 1")
ap.add_argument("-s", "--step", required=False, default=1, type=int,
                help="process every Nth frame; default = 1 (no skipping)")
ap.add_argument("-d", "--delay", required=False, default=0, type=float,
                help="delay time between frames for slo-mo")
ap.add_argument("-v", "--view", required=False, default=1, type=int,
                help="choose whether to view video (1) or not (0)")
ap.add_argument("--debug", action="store_true", help="Log memory usage (RSS) to output CSV for each frame")
ap.add_argument("--ylim", nargs=2, type=int, default=None,
                help="Y-axis range (start end) for brightness averaging, e.g. --ylim 75 470")
ap.add_argument("--xlim", nargs=2, type=int, default=None,
                help="X-axis range (start end) for brightness averaging, e.g. --xlim 20 1210")
ap.add_argument("-t", "--ticks", nargs=4, type=int, default=None,
                help="Four x positions to define three regions (tanks) left to right, e.g. -t 20 450 865 1210")
args = vars(ap.parse_args())

delay = args["delay"]
debug = args["debug"]
gamma = args["gamma"]
black = args["black"]
white = args["white"]
multipleb = args["multipleb"]
initial = args["initial"]
step = args["step"]
view = args["view"]
dir_path = args["path"]

ylim = args["ylim"]
xlim = args["xlim"]

# Check if dir_path exists
if not os.path.exists(dir_path):
    print("The specified directory path does not exist.")
    exit()

# Function to adjust clip
def adjust_clip(image, black=0, white=255):
    zeros = np.array([i * 0 for i in np.arange(0, black)]).astype("uint8")
    whites = np.array([(i * 0) + 255 for i in np.arange(0, 256 - white)]).astype("uint8")
    table = np.array([i + black for i in np.arange(0, white - black)]).astype("uint8")
    table = np.concatenate((zeros, table, whites))
    return cv2.LUT(image, table)

# Function to adjust gamma
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_video(video_path, out_path, initial, step, gamma, black, white, multipleb, view, delay, ylim=None, xlim=None):
    f = open(out_path, "w+")
    ticks = args.get("ticks")
    if ticks:
        f.write("frame,tank1,tank2,tank3,overall\n")
    elif debug:
        f.write("frame,brightness,raw_brightness,memory_MB\n" if not multipleb else "frame," + ",".join([f"b{i+1}" for i in range(len(multipleb))]) + ",raw_brightness,memory_MB\n")
    print(f"Starting brightness analysis for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    print(f"Opened video: {video_path}, success: {cap.isOpened()}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0
    print(f"Total frames reported by OpenCV: {total_frames}")
    print(f"FPS reported by OpenCV: {fps}")
    print(f"Duration (seconds): {duration_sec:.2f}")
    print(f"Duration (minutes): {duration_sec/60:.2f}")
    filename = os.path.basename(video_path)
    buffer = []
    frame_idx = 0
    processed_frames = 0
    from tqdm import tqdm
    import sys
    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            pass
        def set_postfix(self, **kwargs):
            pass
        def update(self, n):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # Use DummyTqdm to suppress tqdm output in notebook
    pbar_cls = DummyTqdm if sys.stdout.isatty() is False else tqdm
    with pbar_cls(desc=f"Processing {filename} (frames)", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            #print(f"Read frame {frame_idx+1}: ret={ret}, frame is None={frame is None}")
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx < initial:
                continue
            if (frame_idx - initial) % step != 0:
                continue
            processed_frames += 1
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                print(f"Warning: Could not convert frame at index {frame_idx} to grayscale. Skipping.")
                continue
            region = gray
            if ylim:
                region = region[ylim[0]:ylim[1], :]
            if xlim:
                region = region[:, xlim[0]:xlim[1]]
            actual_frame_num = frame_idx
            mem_mb = psutil.Process().memory_info().rss / (1024 * 1024) if debug else None
            if ticks:
                # Calculate tank regions
                tank1 = region[:, ticks[0]:ticks[1]]
                tank2 = region[:, ticks[1]:ticks[2]]
                tank3 = region[:, ticks[2]:ticks[3]]
                tank1_brightness = format(tank1.mean(), '.3f')
                tank2_brightness = format(tank2.mean(), '.3f')
                tank3_brightness = format(tank3.mean(), '.3f')
                overall_brightness = format(region.mean(), '.3f')
                line = f"{actual_frame_num},{tank1_brightness},{tank2_brightness},{tank3_brightness},{overall_brightness}"
                buffer.append(line + "\n")
            elif multipleb:
                bright_values = []
                for b in multipleb:
                    clipped = adjust_clip(region, black=b, white=white)
                    adjusted = adjust_gamma(clipped, gamma=gamma)
                    bright = format(adjusted.mean(), '.3f')
                    bright_values.append(bright)
                line = f"{actual_frame_num}," + ",".join(bright_values) + f",{region.mean():.3f}"
                if debug:
                    line += f",{mem_mb:.2f}"
                buffer.append(line + "\n")
            else:
                clipped = adjust_clip(region, black=black, white=white)
                adjusted = adjust_gamma(clipped, gamma=gamma)
                bright = format(adjusted.mean(), '.3f')
                line = f"{actual_frame_num},{bright},{region.mean():.3f}"
                if debug:
                    line += f",{mem_mb:.2f}"
                buffer.append(line + "\n")
            if view == 1:
                cv2.putText(adjusted, str(actual_frame_num), (35, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 180, 10))
                cv2.putText(adjusted, bright_values[0] if multipleb else bright, (175, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 180, 10))
                cv2.putText(adjusted, str(gray.mean()), (375, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 180, 10))
                cv2.imshow('frame', adjusted)
            time.sleep(delay)
            # Use actual FPS from OpenCV for time calculation
            video_time = actual_frame_num / fps if fps > 0 else actual_frame_num / 30.0
            minutes = int(video_time // 60)
            seconds = int(video_time % 60)
            # Only print tqdm update every 1000 actual video frames
            if actual_frame_num % 1000 == 0:
                print(f"Progress: frame {actual_frame_num}, video_time {minutes:02d}:{seconds:02d}")
            pbar.set_postfix(frame=actual_frame_num, time=f"{minutes:02d}:{seconds:02d}")
            pbar.update(1)
            if len(buffer) >= 100:
                f.writelines(buffer)
                f.flush()
                buffer = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if buffer:
            f.writelines(buffer)
            f.flush()
    cap.release()
    f.close()

# Main logic: single-file or directory mode
if args["file"] is None:
    # Process each video file in the directory only if no specific file is given
    for filename in os.listdir(dir_path):
        if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(dir_path, filename)
            video_name = os.path.splitext(filename)[0]
            if args["outfile"]:
                out_path = args["outfile"]
            else:
                out_path = os.path.join(dir_path, f"{video_name}_brightness.csv")
            #print(f"Calling process_video with video_path={video_path}, out_path={out_path}")
            process_video(video_path, out_path, initial, step, gamma, black, white, multipleb, view, delay, ylim=ylim, xlim=xlim)
else:
    # Only process the exact file specified
    filename = args["file"]
    video_path = os.path.join(dir_path, filename)
    if not os.path.exists(video_path):
        print(f"File {video_path} not found.")
    else:
        video_name = os.path.splitext(filename)[0]
        if args["outfile"]:
            out_path = args["outfile"]
        else:
            out_path = os.path.join(dir_path, f"{video_name}_brightness.csv")
        #print(f"Calling process_video with video_path={video_path}, out_path={out_path}")
        process_video(video_path, out_path, initial, step, gamma, black, white, multipleb, view, delay, ylim=ylim, xlim=xlim)

