#!/usr/bin/env bash
#SBATCH --job-name=cawfe_no_fire_eval
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=artifacts/logs/slurm/cawfe_no_fire_eval_%j.out
#SBATCH --error=artifacts/logs/slurm/cawfe_no_fire_eval_%j.err

set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: sbatch $0 <run-dir> <root> <device> <batch-size|default> <num-workers> <overwrite:0|1>" >&2
  exit 2
fi

RUN_DIR="$1"
ROOT="$2"
DEVICE="$3"
BATCH_SIZE="$4"
NUM_WORKERS="$5"
OVERWRITE="$6"
REPO_ROOT="${REPO_ROOT:-/home/mhabibp/FLARE}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"
CONDA_ROOT="${CONDA_ROOT:-/home/mhabibp/anaconda3}"

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_mplconfig

export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python after activating conda environment ${CONDA_ENV}." >&2
  exit 1
fi
if [[ ":${PYTHONPATH:-}:" != *":${REPO_ROOT}:"* ]]; then
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

COMMAND=(
  "${PYTHON_BIN}"
  scripts/recompute_ablation_no_fire_metrics.py
  --run-dir "${RUN_DIR}"
  --root "${ROOT}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
)
if [[ "${BATCH_SIZE}" != "default" ]]; then
  COMMAND+=(--batch-size "${BATCH_SIZE}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  COMMAND+=(--overwrite)
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Run directory: ${RUN_DIR}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Workers: ${NUM_WORKERS}"
"${PYTHON_BIN}" --version
nvidia-smi

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL "${COMMAND[@]}"
