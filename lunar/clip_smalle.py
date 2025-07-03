import cv2
import numpy as np
import argparse
import time
import ntpath
import sys
import os

def parse_time(timestr):
    """Convert MM:SS or SS to float seconds."""
    parts = timestr.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {timestr}")

def path_leaf(path):
    """Extract base filename from full path."""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def clip_smalle(video_path, start_time=0.0, end_time=0.0, delay=0.0, watch=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else total_frames

    if start_frame >= end_frame or end_frame > total_frames:
        raise ValueError("Invalid start/end time or times exceed video duration.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    base = os.path.splitext(path_leaf(video_path))[0]
    output_file = f"{base}_cl_{start_frame}_{end_frame}.mkv"

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = start_frame + frame_count
        if current_frame >= end_frame:
            break

        out.write(frame)

        if watch:
            cv2.imshow('clipping preview', frame)
            if frame_count == 0:
                cv2.moveWindow('clipping preview', 0, 0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

        time.sleep(delay)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Clipped video saved to: {output_file}")

# ---------- CLI wrapper ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip a single video between start and end times (MM:SS or seconds).")
    parser.add_argument("-v", "--video", required=True, help="Path to video file")
    parser.add_argument("-s", "--start", type=str, default="0", help="Start time (MM:SS or seconds), default: 0")
    parser.add_argument("-e", "--end", type=str, default="0", help="End time (MM:SS or seconds), default: end of video")
    parser.add_argument("-d", "--delay", type=float, default=0.0, help="Delay between frames in seconds (default: 0.0)")
    parser.add_argument("-w", "--watch", type=int, default=1, help="1 = preview on, 0 = preview off")

    args = parser.parse_args()
    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end) if args.end != "0" else 0

    clip_smalle(
        video_path=args.video,
        start_time=start_sec,
        end_time=end_sec,
        delay=args.delay,
        watch=bool(args.watch)
    )

