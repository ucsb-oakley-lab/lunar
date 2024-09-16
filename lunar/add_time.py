import pandas as pd
from datetime import datetime, timedelta

def add_time(input_file_name, frame1_time_str, output_file_name, fps=30):
    """
    Adds a new column 'time' to a tab-delimited file that converts video frames into absolute time,
    and writes the result to a new file.
    
    Parameters:
    - input_file_name (str): The name of the input tab-delimited file.
    - frame1_time_str (str): The time corresponding to frame 1 in the format 'YYYY-MM-DD HH:MM:SS'.
    - output_file_name (str): The name of the output file to save the modified data.
    - fps (int): Frames per second, defaults to 30.
    """
    # Read the input file
    df = pd.read_csv(input_file_name, sep='\t')

    # Convert the frame1_time_str to a datetime object
    frame1_time = datetime.strptime(frame1_time_str, '%Y-%m-%d %H:%M:%S')
    
    # Calculate time difference for each frame
    df['time'] = df['frame'].apply(lambda frame: frame1_time + timedelta(seconds=(frame - 1) / fps))

    # Write the modified DataFrame to a new file
    df.to_csv(output_file_name, sep='\t', index=False)

    print(f"New file with absolute time column saved as {output_file_name}")

# Example usage
# add_absolute_time_column_to_file('input_file.txt', '2024-07-22 00:00:00', 'output_file.txt')  # Replace with actual file names


