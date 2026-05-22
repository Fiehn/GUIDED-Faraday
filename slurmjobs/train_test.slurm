#!/bin/bash
#SBATCH --partition=fatq
#SBATCH --time 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/test-%j.out
#SBATCH --job-name=test
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0

# -1) reactivate group permissions
sg 3phaseinsight 

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

module load Python/3.11.3-GCCcore-12.3.0

# 1)
python3 -c "import os; d='/groups/3phaseinsight/code/Data/processed_data/second_batch/individual_sm_datasets/raw_profiles/'; print('listdir:', len(os.listdir(d)) if os.access(d, os.X_OK) else 'Permission denied');"
