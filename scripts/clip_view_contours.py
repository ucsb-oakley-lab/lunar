import argparse
import pandas as pd
import cv2
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Create a video clip of contours with circles from a tab-separated file.")
    parser.add_argument("-i", "--input", required=True, help="Input tab-separated file with contour info.")
    parser.add_argument("-s", "--start", type=int, required=True, help="Start row (1-based index, inclusive)")
    parser.add_argument("-e", "--end", type=int, required=True, help="End row (1-based index, inclusive)")
    parser.add_argument("-p", "--prefix", default="clip", help="Output file prefix (default: clip)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video (default: 10)")
    parser.add_argument("--keepdark", action="store_true", help="Include all frames in the range, even those without contours.")
    return parser.parse_args()

def draw_contour_circle(frame, cX, cY, area):
    # Estimate radius from area (area = pi * r^2), scale up by 1.5
    radius = int(np.sqrt(area / np.pi) * 1.5)
    color = (0, 255, 255)  # Yellow in BGR
    thickness = 2
    cv2.circle(frame, (int(cX), int(cY)), radius, color, thickness)
    return frame

def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep='\t')
    # Ensure rows are ordered by frame
    df = df.sort_values('frame')
    # Select specified range (convert to 0-based)
    start_idx = args.start - 1
    end_idx = args.end
    rows = df.iloc[start_idx:end_idx]
    out_name = f"{args.prefix}_{args.start}_{args.end}.mp4"
    frames = []
    if args.keepdark:
        # Group by video, process each video separately
        grouped = rows.groupby('video')
        for video_path, group in grouped:
            min_frame = int(group['frame'].min())
            max_frame = int(group['frame'].max())
            # Build a lookup for contours in this video
            contour_dict = {int(row['frame']): row for _, row in group.iterrows()}
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue
            for fnum in range(min_frame, max_frame + 1):
                print(f"Processing video: {video_path}, frame: {fnum}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                ret, frame = cap.read()
                if not ret:
                    print(f"Could not read frame {fnum} from {video_path}")
                    continue
                if fnum in contour_dict:
                    row = contour_dict[fnum]
                    frame = draw_contour_circle(frame, row['cX'], row['cY'], row['area'])
                frames.append(frame)
            cap.release()
    else:
        last_video = None
        cap = None
        for idx, row in rows.iterrows():
            video_path = row['video']
            frame_num = int(row['frame'])
            cX = row['cX']
            cY = row['cY']
            area = row['area']
            print(f"Processing row {idx+1}: video={video_path}, frame={frame_num}, cX={cX}, cY={cY}, area={area}")
            # Open video if changed
            if last_video != video_path:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    continue
                last_video = video_path
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame {frame_num} from {video_path}")
                continue
            # Draw circle
            frame = draw_contour_circle(frame, cX, cY, area)
            frames.append(frame)
        if cap is not None:
            cap.release()
    if not frames:
        print("No frames extracted. Exiting.")
        return
    # Get frame size
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_name, fourcc, args.fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved clip to {out_name}")

if __name__ == "__main__":
    main()
