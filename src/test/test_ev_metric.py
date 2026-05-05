import numpy as np
import pandas as pd
from src.evaluation.ev_detection import detect_platos_unified_dP2_std
from src.evaluation.eval_framework import EVConditionAccuracy

def run_tests():
    print("=========================================")
    print("1. Testing EV Detection Logic Directly")
    print("=========================================")
    
    # Create 96 steps of 15min intervals (24 hours) for direct testing
    dates = pd.date_range("2024-01-01", periods=96, freq="15min")
    l1 = np.random.normal(500, 10, 96)
    l2 = np.random.normal(500, 10, 96)
    l3 = np.random.normal(500, 10, 96)

    # Inject a 3-phase EV charging session from step 20 to 40 (5 hours)
    l1[20:40] += 4000
    l2[20:40] += 4000
    l3[20:40] += 4000

    df = pd.DataFrame({
        "active_power_p14_l1_test": l1,
        "active_power_p14_l2_test": l2,
        "active_power_p14_l3_test": l3,
    }, index=dates)

    events = detect_platos_unified_dP2_std(df, "test", interval_minutes=15)
    print("Detected events for injected 3-phase EV:")
    print(events[['start_time', 'end_time', 'duration_hours', 'phase_type']])


    print("\n=========================================")
    print("2. Testing EVConditionAccuracy Metric (Phased)")
    print("=========================================")
    # Simulate metric's unpack format.
    # Using the default (interval_minutes=60, window_size=24) which expects hourly data
    
    # Sample 0: 3-phase EV in kW (max < 100, metric will scale by 1000)
    sample0 = np.ones((24, 3)) * 0.5 
    sample0[8:14, :] += 4.0  # 3-phase EV
    
    # Sample 1: 1-phase EV in kW
    sample1 = np.ones((24, 3)) * 0.5
    sample1[12:18, 0] += 5.0 # 1-phase EV on L1
    
    # Sample 2: No EV
    sample2 = np.ones((24, 3)) * 0.5 

    synthetic_data_phased = np.stack([
        sample0.flatten(),
        sample1.flatten(),
        sample2.flatten()
    ])

    conditions = np.array([
        [0, 0, 0, 1], # Expects 3ph (Matches sample 0)
        [0, 1, 0, 0], # Expects 1ph (Matches sample 1)
        [1, 0, 0, 0], # Expects No EV (Matches sample 2)
    ])

    metric = EVConditionAccuracy(window_size=24, interval_minutes=60)
    metric.synthetic_data = synthetic_data_phased
    metric.synthetic_conditions = conditions

    accuracy = metric.compute()
    print(f"Total Samples Tested (Phased): {len(synthetic_data_phased)}")
    print(f"Computed Accuracy: {accuracy * 100:.2f}%")


    print("\n=========================================")
    print("3. Testing EVConditionAccuracy Metric (Aggregated)")
    print("=========================================")
    
    # Aggregated profile sums the phases.
    # Sample 0: 3-phase EV in kW (12kW jump total)
    sample_agg0 = np.ones((24, 1)) * 1.5 
    sample_agg0[8:14, 0] += 12.0  # 3-phase EV equivalent (12000W)
    
    # Sample 1: 1-phase EV in kW (5kW jump total)
    sample_agg1 = np.ones((24, 1)) * 1.5
    sample_agg1[12:18, 0] += 5.0  # 1-phase EV equivalent
    
    # Sample 2: No EV (no jump)
    sample_agg2 = np.ones((24, 1)) * 1.5 
    
    synthetic_data_agg = np.stack([
        sample_agg0.flatten(),
        sample_agg1.flatten(),
        sample_agg2.flatten()
    ])
    
    # The expected labels stay the same in formulation
    metric_agg = EVConditionAccuracy(window_size=24, interval_minutes=60)
    metric_agg.synthetic_data = synthetic_data_agg
    metric_agg.synthetic_conditions = conditions

    accuracy_agg = metric_agg.compute()
    print(f"Total Samples Tested (Aggregated): {len(synthetic_data_agg)}")
    print(f"Computed Accuracy: {accuracy_agg * 100:.2f}%")
    print("=========================================")

if __name__ == "__main__":
    run_tests()
