import pandas as pd
import numpy as np
import json

def count_zipcodes(json_path="/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned/Cleaned_SM_Topology_mapping.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    zipcode_counts = {}
    
    for key, value in data.items():
        zipcode = value["Zip Code"]
        if zipcode in zipcode_counts:
            zipcode_counts[zipcode] += 1
        else:
            zipcode_counts[zipcode] = 1
    
    # print total number of zipcodes
    print(f"Total number of unique zipcodes: {len(zipcode_counts)}")
    print(f"Total number of samples: {sum(zipcode_counts.values())}")
    return zipcode_counts



if __name__ == "__main__":
    zipcode_counts = count_zipcodes()
    print(zipcode_counts)