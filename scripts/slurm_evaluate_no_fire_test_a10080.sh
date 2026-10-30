#!/usr/bin/env bash
#SBATCH --job-name=no_fire_test
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=artifacts/logs/slurm/no_fire_test_%x_%j.out
#SBATCH --error=artifacts/logs/slurm/no_fire_test_%x_%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: sbatch $0 <baseline|GA_Q2> <42|123|2026>" >&2
  exit 2
fi

MODEL="$1"
SEED="$2"
REPO_ROOT="${REPO_ROOT:-/home/mhabibp/FLARE}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_no_fire_test_mplconfig
export MPLCONFIGDIR=/tmp/mhabibp_no_fire_test_mplconfig
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python after activating ${CONDA_ENV}." >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "NO-FIRE TEST EVALUATION"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Model: ${MODEL}"
echo "Seed: ${SEED}"
echo "Frozen checkpoints only; no training is permitted."
"${PYTHON_BIN}" --version
nvidia-smi

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/evaluate_no_fire_test.py \
  --models "${MODEL}" --seeds "${SEED}" --evaluate-only
