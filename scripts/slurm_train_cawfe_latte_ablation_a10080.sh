#!/usr/bin/env bash
#SBATCH --job-name=cawfe_latte_ablation
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=4:00:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=artifacts/logs/slurm/cawfe_latte_ablation_%j.out
#SBATCH --error=artifacts/logs/slurm/cawfe_latte_ablation_%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch $0 <registered-ablation-name>" >&2
  exit 2
fi

ABLATION="$1"
REPO_ROOT="${REPO_ROOT:-/home/mhabibp/FLARE}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"
RUN_ID="${RUN_ID:-slurm${SLURM_JOB_ID:-local}}"

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_mplconfig

export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export FIRE_FORECASTING_PROGRESS_BAR=1
export FIRE_FORECASTING_PROGRESS_PERCENT=0
export FIRE_FORECASTING_TIMING_LOG_EVERY_N_BATCHES=50
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python after activating conda environment ${CONDA_ENV}." >&2
  exit 1
fi
if [[ ":${PYTHONPATH:-}:" != *":${REPO_ROOT}:"* ]]; then
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

CONFIG_PATH="$("${PYTHON_BIN}" scripts/run_cawfe_latte_ablation.py "${ABLATION}" --print-config)"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Ablation: ${ABLATION}"
echo "Config: ${CONFIG_PATH}"
echo "Run ID: ${RUN_ID}"
echo "Conda env: ${CONDA_ENV}"
"${PYTHON_BIN}" --version
nvidia-smi

"${PYTHON_BIN}" scripts/check_cawfe_latte_ablation_configs.py

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL   /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}"   scripts/sanity_check_project.py   --config "${CONFIG_PATH}"   --batch_size 2   --num_workers 0   --deep

echo "Starting 10-epoch screening training; the runner will reload best_model.pt and evaluate the full validation split before exit."
srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL   /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}"   scripts/run_cawfe_latte_ablation.py "${ABLATION}" --run-id "${RUN_ID}"
echo "Completed training and automatic full validation for ${ABLATION}."
