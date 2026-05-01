from .eval_framework import SyntheticMetric
import numpy as np
import matplotlib.pyplot as plt



class PeakAnalysis(SyntheticMetric):

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "Peak Analysis"
    
    def plot_load_duration_curve(self) -> None:
        # We sort the data in descending order to create a duration curve
        sort_real_data = np.sort(self.real_data.flatten())[::-1]
        sort_synthetic_data = np.sort(self.synthetic_data.flatten())[::-1]

        if len(sort_real_data) != len(sort_synthetic_data):
            min_len = min(len(sort_real_data), len(sort_synthetic_data))
            sort_real_data = sort_real_data[:min_len]
            sort_synthetic_data = sort_synthetic_data[:min_len]
        # Create the x-axis values (time steps)
        x = np.arange(1, len(sort_real_data) + 1)
        # Plotting the load duration curve
        plt.figure(figsize=(10, 6))
        plt.plot(x, sort_real_data, label='Real Data', color='blue')
        plt.plot(x, sort_synthetic_data, label='Synthetic Data', color='orange')
        plt.title('Load Duration Curve')
        plt.xlabel('Time Steps')
        plt.ylabel('Load')
        plt.legend()
        plt.grid()
        save_path = self.output_path[:-17]
        plt.savefig(f"{save_path}/load_duration_curve.png")
        plt.close()


    def compute(self):
        self.plot_load_duration_curve()

        
        # TODO: actual metrics