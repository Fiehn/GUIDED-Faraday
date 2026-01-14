import numpy as np
import pandas as pd
import os

class RawData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = None

    def load_data(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found in directory {self.data_dir}")
        data = pd.read_csv(file_path)
        return data
    def preprocess_data(self, data):
        # The columns are: meter_number, timestamp_dst, timestamp_utc, hourly_energy_consumption, hourly_energy_production
        # The meter_number is a unique identifier for each meter, this will be used for indexing
        data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'])
        data.set_index(['meter_number', 'timestamp_utc'], inplace=True)
        data.sort_index(inplace=True)

        # 0 value imputation for consumption and production
        data['hourly_energy_consumption'].replace(0, np.nan, inplace=True)
        data['hourly_energy_production'].replace(0, np.nan, inplace=True)

        return data
    def get_data(self, filename):
        raw_data = self.load_data(filename)
        self.data = self.preprocess_data(raw_data)
        return self.data
    def get_all_meter_numbers(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first using get_data method.")
        return self.data.index.get_level_values('meter_number').unique().tolist()
    def get_data_by_meter(self, meter_number):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first using get_data method.")
        return self.data.xs(meter_number, level='meter_number')

    def examine_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first using get_data method.")
        print("Data Info:")
        print(self.data.info())
        print("\nData Description:")
        print(self.data.describe())
        print("\nMissing Values:")
        print(self.data.isnull().sum())

if __name__ == "__main__":
    data_dir = "data"
    filename = "hourly_measurements_2023_12.csv"
    raw_data_loader = RawData(data_dir)
    data = raw_data_loader.get_data(filename)
    print("Data loaded and preprocessed successfully.")
    print(data.head())
    raw_data_loader.examine_data()

    # How many unique meter numbers are there?
    unique_meters = raw_data_loader.get_all_meter_numbers()
    print(f"Number of unique meter numbers: {len(unique_meters)}")

    # Get data for a specific meter number
    sample_meter_number = unique_meters[0]
    meter_data = raw_data_loader.get_data_by_meter(sample_meter_number)
    print(f"Data for meter number {sample_meter_number}:")
    print(meter_data.head())
    # Plot some consumption for a random 24 hours
    import matplotlib.pyplot as plt
    sample_data = meter_data.head(24)
    plt.figure(figsize=(10, 5))
    plt.plot(sample_data.index, sample_data['hourly_energy_consumption'], marker='o')
    plt.title(f'Hourly Energy Consumption for Meter {sample_meter_number} (First 24 Hours)')
    plt.xlabel('Timestamp (UTC)')
    plt.ylabel('Hourly Energy Consumption')
    plt.grid()
    plt.show()

