#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/train_all_models-%j.out
#SBATCH --job-name=train_all_models
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

# 1) run training
python -u src/train.py --skip-validation --config "config/tests/baseline_mean.yaml"
python -u src/train.py --skip-validation --config "config/tests/baseline_quantile.yaml"
python -u src/train.py --skip-validation --config "config/tests/cvae.yaml"
python -u src/train.py --skip-validation --config "config/tests/guide.yaml"

python -u src/train.py --skip-validation --config "config/tests/art.yaml"
python -u src/train.py --skip-validation --config "config/tests/diffusion.yaml"

python -u src/train.py --skip-validation --config "config/tests/guide_cnn.yaml"
python -u src/train.py --skip-validation --config "config/tests/guide_hierarchy.yaml"
python -u src/train.py --skip-validation --config "config/tests/guide_vamp.yaml"
