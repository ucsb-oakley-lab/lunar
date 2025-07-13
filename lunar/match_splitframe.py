# lunar/match_splitframe.py

import pandas as pd
import numpy as np
import argparse

def match_splitframe_contours(input_file, output_file, distance_x=200, distance_y=200, frame_width=3840):
    """
    Matches contours detected on left and right sides of a frame split down the middle.

    Adds a 'match' column:
    - 'N': matched on both sides
    - 'L': only on left
    - 'R': only on right
    """

    df = pd.read_csv(input_file, delimiter='\t')

    mid_x = frame_width // 2
    df['side'] = df['cX'].apply(lambda x: 'left' if x < mid_x else 'right')
    df['match'] = ''

    def find_closest_pairs(left, right, left_indices, right_indices, distance_x, distance_y):
        pairs = []
        used_right = set()
        for l_val, l_idx in zip(left, left_indices):
            closest_idx = None
            closest_dist = float('inf')
            for r_val, r_idx in zip(right, right_indices):
                if r_idx in used_right:
                    continue
                x_diff = abs(l_val[0] - r_val[0])
                y_diff = abs(l_val[1] - r_val[1])
                if x_diff <= distance_x and y_diff <= distance_y:
                    dist = np.sqrt(x_diff**2 + y_diff**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = r_idx
                        closest_r_val = r_val
            if closest_idx is not None:
                pairs.append((l_val, closest_r_val, l_idx, closest_idx))
                used_right.add(closest_idx)
        return pairs

    for frame_num, group in df.groupby('frame'):
        group = group.copy()

        # Normalize cX so that right side is shifted left by mid_x
        group['cXnorm'] = group['cX']
        group.loc[group['side'] == 'right', 'cXnorm'] -= mid_x

        left_df = group[group['side'] == 'left'][['cXnorm', 'cY']]
        right_df = group[group['side'] == 'right'][['cXnorm', 'cY']]
        left_indices = left_df.index
        right_indices = right_df.index

        left_vals = left_df.values
        right_vals = right_df.values

        pairs = find_closest_pairs(left_vals, right_vals, left_indices, right_indices, distance_x, distance_y)

        matched_left = set()
        matched_right = set()

        for (l_val, r_val, l_idx, r_idx) in pairs:
            df.at[l_idx, 'match'] = 'N'
            df.at[r_idx, 'match'] = 'N'
            matched_left.add(l_idx)
            matched_right.add(r_idx)

        for l_idx in left_indices:
            if l_idx not in matched_left:
                df.at[l_idx, 'match'] = 'L'
        for r_idx in right_indices:
            if r_idx not in matched_right:
                df.at[r_idx, 'match'] = 'R'

    df.drop(columns=['side'], inplace=True)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match contours from left and right views in a split-frame video.")
    parser.add_argument("input_file", help="Input tab-delimited file")
    parser.add_argument("output_file", help="Output file with match column")
    parser.add_argument("--distance_x", type=int, default=200, help="Max horizontal distance for a match (default: 200)")
    parser.add_argument("--distance_y", type=int, default=200, help="Max vertical distance for a match (default: 200)")
    parser.add_argument("--frame_width", type=int, default=3840, help="Total frame width of the video (default: 3840)")

    args = parser.parse_args()
    match_splitframe_contours(
        input_file=args.input_file,
        output_file=args.output_file,
        distance_x=args.distance_x,
        distance_y=args.distance_y,
        frame_width=args.frame_width
    )

