#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --time 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --output=slurmjobs/slurm-%j.out
#SBATCH --job-name=env_cuda_test
#SBATCH --mail-user=s204617@dtu.dk
#SBATCH --mail-type=END,FAIL

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0

# 0) Start clean
rm -rf .venv-cu12-py311
python3.11 -m venv .venv-cu12-py311
source .venv-cu12-py311/bin/activate

# 1) Modern pip tooling
python -m pip install --upgrade pip setuptools wheel

# 2) Install base deps from PyPI (NOT torch index)
python -m pip install --prefer-binary -r requirements_hpc_base.txt

# 3) Install CUDA 12 torch, but keep PyPI available for deps
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# 4) Verify
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
