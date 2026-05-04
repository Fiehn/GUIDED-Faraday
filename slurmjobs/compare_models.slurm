#!/bin/bash
#SBATCH --partition=workq
#SBATCH --time 3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/compare_models-%j.out
#SBATCH --job-name=test
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

# 1)
python3 src/evaluation/compare_models.py






