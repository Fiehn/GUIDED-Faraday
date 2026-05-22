#!/bin/bash
#SBATCH --partition=fatq
#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurmjobs/outputs/test2-%j.out
#SBATCH --job-name=phased
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

sg 3phaseinsight

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0

# 0) Load virtual environment
source .venv-cu12-py311/bin/activate

python -u src/train.py --skip-validation --config "config/tests/phased/baseline_average.yaml"

python -u src/train.py --skip-validation --config "config/tests/phased/baseline_quantile.yaml"

python -u src/train.py --skip-validation --config "config/tests/phased/art.yaml"

python -u src/train.py --skip-validation --config "config/tests/phased/diffusion.yaml"

python -u src/train.py --skip-validation --config "config/tests/phased/phase_splitter.yaml"

python -u src/train.py --skip-validation --config "config/tests/phased/sequential.yaml"

python -u src/train.py --skip-validation --config "config/tests/phased/set_guide_cnn.yaml"

python -u src/train.py --skip-validation --config "config/tests/phased/set_guide.yaml"

