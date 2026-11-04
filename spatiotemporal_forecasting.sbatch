#!/bin/bash
#SBATCH --job-name=taxi_stmgcn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=stmgcn_%j.out
#SBATCH --error=stmgcn_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Override TAXI_PYTHON when using a different Avon virtual environment.
TAXI_PYTHON="${TAXI_PYTHON:-$PWD/.venv-avon-new/bin/python}"
export MPLBACKEND=Agg
srun "$TAXI_PYTHON" -u spatiotemporal_forecasting.py --stage all --device cuda
