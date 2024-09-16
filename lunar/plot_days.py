import pandas as pd
import matplotlib.pyplot as plt

def plot_days(file_name, x_axis='frame'):
    """
    Plot days with average contours against a chosen x-axis ('frame' or 'time').

    Parameters:
    - file_name: str, the name of the input file containing the data.
    - x_axis: str, the column name to use for the x-axis ('frame' or 'time').

    Returns:
    - None
    """
    # Read the tab-delimited file into a DataFrame
    data = pd.read_csv(file_name, delimiter='\t')
    
    # Ensure required columns exist, including the chosen x-axis
    required_columns = {'date', 'average_contours', 'kclusters', x_axis}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")
    
    # Sort data by 'date' and the chosen x-axis to ensure proper plotting order
    data = data.sort_values(by=['date', x_axis])
    
    # Get unique dates
    unique_dates = data['date'].unique()
    
    # Define colors for different kclusters
    kcluster_colors = {1: 'lightgrey', 2: 'darkgrey'}
    
    # Determine the y-axis range based on 'average_contours'
    y_min = data['average_contours'].min()
    y_max = data['average_contours'].max()
    
    # Create subplots with a wide size and smaller vertical size for panels
    fig, axes = plt.subplots(len(unique_dates), 1, figsize=(12, 1.5 * len(unique_dates)), sharex=True)
    
    if len(unique_dates) == 1:  # If there's only one unique date, make axes iterable
        axes = [axes]
    
    for ax, date in zip(axes, unique_dates):
        # Filter data for the current date
        date_data = data[data['date'] == date]
        
        # Convert relevant columns to numpy arrays for plotting
        x_values = date_data[x_axis].to_numpy()
        average_contours = date_data['average_contours'].to_numpy()
        
        # Plot average_contours against the chosen x-axis using scatter with smaller points
        ax.scatter(x_values, average_contours, color='black', s=2)  # 's' is set to 2 for smaller points
        
        # Group by consecutive kcluster values and plot the shaded areas
        current_kcluster = date_data.iloc[0]['kclusters']
        start_x = date_data.iloc[0][x_axis]
        
        for i in range(1, len(date_data)):
            x_val = date_data.iloc[i][x_axis]
            kcluster = date_data.iloc[i]['kclusters']
            
            if kcluster != current_kcluster or i == len(date_data) - 1:
                # End of the current segment
                end_x = x_val if kcluster == current_kcluster else date_data.iloc[i - 1][x_axis]
                if current_kcluster in kcluster_colors:  # Only shade for kcluster values 1 and 2
                    ax.fill_betweenx([0, y_max], start_x, end_x, color=kcluster_colors[current_kcluster], alpha=0.5)
                
                # Start a new segment
                current_kcluster = kcluster
                start_x = x_val
        
        # Set title, limits, and labels
        ax.set_title(f"Date: {date}")
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('Average Contours')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel(x_axis.capitalize())
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_days('data.txt', x_axis='time')



def plot_days_old(file_name):
    # Read the tab-delimited file into a DataFrame
    data = pd.read_csv(file_name, delimiter='\t')
    
    # Ensure 'date', 'frame', 'average_contours', and 'kclusters' columns exist
    required_columns = {'date', 'frame', 'average_contours', 'kclusters'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")
    
    # Sort data by 'date' and 'frame' to ensure proper plotting order
    data = data.sort_values(by=['date', 'frame'])
    
    # Get unique dates
    unique_dates = data['date'].unique()
    
    # Define colors for different kclusters
    kcluster_colors = {1: 'lightgrey', 2: 'darkgrey'}  # Define colors for different kclusters
    
    # Determine the y-axis range based on 'average_contours'
    y_min = data['average_contours'].min()
    y_max = data['average_contours'].max()
    
    # Create subplots with a wide size and smaller vertical size for panels
    fig, axes = plt.subplots(len(unique_dates), 1, figsize=(12, 1.5 * len(unique_dates)), sharex=True)  # Adjusted height to 1.5 per panel
    
    if len(unique_dates) == 1:  # If there's only one unique date, make axes iterable
        axes = [axes]
    
    for ax, date in zip(axes, unique_dates):
        # Filter data for the current date
        date_data = data[data['date'] == date]
        
        # Convert relevant columns to numpy arrays for plotting
        frames = date_data['frame'].to_numpy()
        average_contours = date_data['average_contours'].to_numpy()
        
        # Plot average_contours against frame using scatter with smaller points
        ax.scatter(frames, average_contours, color='black', s=2)  # 's' is set to 2 for smaller points
        
        # Group by consecutive kcluster values and plot the shaded areas
        current_kcluster = date_data.iloc[0]['kclusters']
        start_frame = date_data.iloc[0]['frame']
        
        for i in range(1, len(date_data)):
            frame = date_data.iloc[i]['frame']
            kcluster = date_data.iloc[i]['kclusters']
            
            if kcluster != current_kcluster or i == len(date_data) - 1:
                # End of the current segment
                end_frame = frame if kcluster == current_kcluster else date_data.iloc[i - 1]['frame']
                if current_kcluster in kcluster_colors:  # Only shade for kcluster values 1 and 2
                    ax.fill_betweenx([0, y_max], start_frame, end_frame, color=kcluster_colors[current_kcluster], alpha=0.5)
                
                # Start a new segment
                current_kcluster = kcluster
                start_frame = frame
        
        # Set title, limits, and labels
        ax.set_title(f"Date: {date}")
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('Average Contours')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel('Frame')
    plt.tight_layout()
    plt.show()

