#!/usr/bin/env python3

import cv2
import argparse
import matplotlib.pyplot as plt

def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Frame {frame_number} not found in video {video_path}")
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (Matplotlib)

def show_frame_with_lines(frame, x_positions, save_path=None, cameras=1):
    height, width, _ = frame.shape
    plt.imshow(frame)
    if cameras == 2:
        # For 2-camera mode, assume the frame is double width, so lines are mirrored
        for x in x_positions:
            plt.axvline(x=x, color='red')
            plt.axvline(x=x + width // 2, color='blue')
    else:
        for x in x_positions:
            plt.axvline(x=x, color='red')
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and display a frame from a video with vertical lines")
    parser.add_argument('-v', '--video', type=str, required=True, help="Input video file path")
    parser.add_argument('-f', '--frame', type=int, required=True, help="Frame number to extract")
    parser.add_argument('-t', '--ticks', type=int, nargs='+', required=True, help="X-axis positions for vertical lines")
    parser.add_argument('-o', '--output', type=str, default=None, help="Optional output image file path")
    parser.add_argument('--cameras', type=int, choices=[1,2], default=1, help="Number of cameras in video (1 or 2). Default is 1.")

    args = parser.parse_args()

    frame = extract_frame(args.video, args.frame)
    show_frame_with_lines(frame, args.ticks, save_path=args.output, cameras=args.cameras)

