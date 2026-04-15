#!/bin/bash
#SBATCH --partition=workq
#SBATCH --time 03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/subset_creator-%j.out
#SBATCH --job-name=create_subset_data
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

# 1) run training user profiler training
python -u src/data/usable/subset_creator.py

