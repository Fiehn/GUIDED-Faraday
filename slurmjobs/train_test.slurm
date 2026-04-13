#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --time 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/train_test-%j.out
#SBATCH --job-name=train_test
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

# 1) run training
python src/train.py --skip-validation --config "config/config_test.yaml"
