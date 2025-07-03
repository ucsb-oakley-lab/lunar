import cv2
import numpy as np
import time
import argparse

def adjust_clip(image, black=0, white=255):
    zeros = np.zeros(black, dtype="uint8")
    table = np.arange(black, white, dtype="uint8")
    whites = np.full(256 - white, 255, dtype="uint8")
    full_table = np.concatenate((zeros, table, whites))
    return cv2.LUT(image, full_table)

def adjust_gamma(image, gamma=1.0):
    gamma = max(gamma, 0.01)
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255 for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

def play_smalle_video(video_path, gamma=0.8, black=110, white=255, start=1, delay=0.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frametext = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frametext += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clipped = adjust_clip(gray, black=black, white=white)
        adjusted = adjust_gamma(clipped, gamma=gamma)

        height, width = adjusted.shape
        winname = f"frame ({width}x{height})"
        cv2.putText(adjusted, f"Frame {frametext + start}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,), 1)
        cv2.imshow(winname, adjusted)

        if frametext == 1:
            cv2.moveWindow(winname, 0, 0)

        time.sleep(delay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# ---------- CLI Block ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a single video with adjustments")
    parser.add_argument("-v", "--video", required=True, help="Path to video file")
    parser.add_argument("-g", "--gamma", type=float, default=0.8, help="Gamma correction (default: 0.8)")
    parser.add_argument("-b", "--black", type=int, default=110, help="Black threshold (default: 110)")
    parser.add_argument("-w", "--white", type=int, default=255, help="White threshold (default: 255)")
    parser.add_argument("-s", "--start", type=int, default=1, help="Start frame (default: 1)")
    parser.add_argument("-d", "--delay", type=float, default=0.0, help="Delay between frames in seconds (default: 0.0)")
    args = parser.parse_args()

    play_smalle_video(
        video_path=args.video,
        gamma=args.gamma,
        black=args.black,
        white=args.white,
        start=args.start,
        delay=args.delay
    )

