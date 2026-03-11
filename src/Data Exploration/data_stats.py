import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
class PlotColors:
    """Class to define color schemes for plots."""
    def __init__(self):
        self.colors = {
            'blue': '#1f77b4',
            'orange': '#ff7f0e',
            'green': '#2ca02c',
            'red': '#d62728',
            'purple': '#9467bd',
            'brown': '#8c564b',
            'pink': '#e377c2',
            'gray': '#7f7f7f',
            'olive': '#bcbd22',
            'cyan': '#17becf'
        }

    def get_color(self, color_name):
        """Return the hex code for a given color name."""
        return self.colors.get(color_name, '#000000')  # Default to black if color not found
    
    def get_all_colors(self):
        """Return the entire color dictionary."""
        return self.colors
    
    def get_colors(self, amount):
        """Return a list of colors based on the specified amount."""
        color_list = list(self.colors.values())
        if amount <= len(color_list):
            return color_list[:amount]
        else:
            # If more colors are needed than available, repeat the color list
            return (color_list * (amount // len(color_list) + 1))[:amount]
    
    def sns_color_palette(self, amount):
        """Return a seaborn color palette based on the specified amount."""
        return sns.color_palette(self.get_colors(amount))


# Get two random meter ids entire month data and return plots for the same
def plot_random_meter_data(df, num_meters=2):
    """Plot data for a specified number of random meter IDs."""
    plot_colors = PlotColors()
    unique_meters = df['meter_id'].unique()
    
    if num_meters > len(unique_meters):
        raise ValueError("Number of meters requested exceeds the number of unique meters in the dataset.")
    
    random_meters = pd.Series(unique_meters).sample(num_meters, random_state=42).tolist()
    
    # Create a figure with subplots for each meter (the top for full meter data and the bottom for daily average)
    fig, axes = plt.subplots(num_meters, 2, figsize=(15,
                                                5 * num_meters), sharex=True)
    for i, meter_id in enumerate(random_meters):
        meter_data = df[df['meter_id'] == meter_id]
        
        # Plot full meter data
        sns.lineplot(ax=axes[i, 0], x='timestamp', y='value', data=meter_data, color=plot_colors.get_color('blue'))
        axes[i, 0].set_title(f'Meter ID: {meter_id} - Full Data')
        axes[i, 0].set_xlabel('Timestamp')
        axes[i, 0].set_ylabel('Value')
        
        # Plot daily average
        daily_avg = meter_data.resample('D', on='timestamp')['value'].mean().reset_index()
        sns.lineplot(ax=axes[i, 1], x='timestamp', y='value', data=daily_avg, color=plot_colors.get_color('orange'))
        axes[i, 1].set_title(f'Meter ID: {meter_id} - Daily Average')
        axes[i, 1].set_xlabel('Timestamp')
        axes[i, 1].set_ylabel('Average Value')

    plt.tight_layout()
    plt.savefig('src//Data Exploration/random_meter_data_plots.png')
    plt.show()

def plot_statisical_measures(df):
    """Plot statistical measures for the dataset."""
    plot_colors = PlotColors()
    
    # Calculate statistical measures
    mean_value = df['value'].mean()
    median_value = df['value'].median()
    std_value = df['value'].std()
    
    # Create a bar plot for the statistical measures
    measures = ['Mean', 'Median', 'Standard Deviation']
    values = [mean_value, median_value, std_value]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=measures, y=values, palette=plot_colors.sns_color_palette(len(measures)))
    plt.title('Statistical Measures of Meter Values')
    plt.ylabel('Value')
    
    plt.savefig('src//Data Exploration/statistical_measures_plot.png')
    plt.show()

def plot_value_distribution(df):
    """Plot the distribution of meter values."""
    plot_colors = PlotColors()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['value'], bins=30, kde=True, color=plot_colors.get_color('green'))
    plt.title('Distribution of Meter Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.savefig('src//Data Exploration/value_distribution_plot.png')
    plt.show()

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/hourly_measurements_2023_12.csv', parse_dates=['timestamp_utc'])
    df.rename(columns={'hourly_energy_consumption': 'value'}, inplace=True) 

    # Plot random meter data
    plot_random_meter_data(df, num_meters=2)
    
    # Plot statistical measures
    plot_statisical_measures(df)
    
    # Plot value distribution
    plot_value_distribution(df)