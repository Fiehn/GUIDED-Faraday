#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/train_all_non_normalizer-%j.out
#SBATCH --job-name=train_all_models_non_normalized
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

# 1) run training

python -u src/train.py --skip-validation --config "config/tests/agg/monster.yaml"
python -u src/train.py --skip-validation --config "config/tests/agg/guide_cnn.yaml"
python -u src/train.py --skip-validation --config "config/tests/agg/diffusion2.yaml"
python -u src/train.py --skip-validation --config "config/tests/agg/monster2.yaml"
