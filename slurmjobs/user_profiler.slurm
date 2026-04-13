#!/bin/bash
#SBATCH --partition=workq
#SBATCH --time 05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/user_profiler-%j.out
#SBATCH --job-name=user_profiler_training_non_phased
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

# 1) run training user profiler training
python -u src/train.py --skip-validation --fit_user_profiler_only --config "config/config_test.yaml"
