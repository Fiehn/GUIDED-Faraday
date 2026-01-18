import numpy as np
import torch

from dataloader import SmartMeterDataModule, WindowSpec

def preview_loader(name, loader, n_batches=1):
    print(f"\n=== {name} ===")

    # dataset size (number of windows)
    ds = loader.dataset
    print("num window-samples:", len(ds))

    # peek at first batch
    for i, x in enumerate(loader):
        print(f"batch {i}: x {tuple(x.shape)}")
        print("x[0, :3, :]:\n", x[0, :3, :])   # first 3 timesteps of first sample
        break

def main():
    dm = SmartMeterDataModule(
        data_dir="/Users/clarareginekold/Desktop/dtu/THESIS/Thesis/data/hourly_measurements_package1",
        filename="hourly_measurements_2023_9.csv",

        meter_col="meter_number",
        timestamp_dst_col="timestamp_dst",
        timestamp_utc_col="timestamp_utc",

        feature_cols=("hourly_energy_consumption", "hourly_energy_production"),
        window=WindowSpec(input_len=24, stride=1),

        batch_size=16,
        num_workers=0,
        shuffle_train=False,
        seed=0,
        train_frac=0.8,
        val_frac=0.1,
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()

    preview_loader("TRAIN", train_loader)
    preview_loader("VAL", val_loader)
    preview_loader("TEST", test_loader)

if __name__ == "__main__":
    main()

# output: shuffled meter IDs, then assigned ~80% of meters to train, ~10% val, ~10% test.
# no meter appears in more than one split. 
# each window has 24 hours and 2 features (consumption, production). batch_size=16, so the DataLoader gives 16 windows at once