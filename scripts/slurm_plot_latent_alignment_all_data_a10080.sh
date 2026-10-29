#!/usr/bin/env bash
#SBATCH --job-name=flare_latent_all
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm/flare_latent_all_%j.out
#SBATCH --error=artifacts/logs/slurm/flare_latent_all_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mhabibp/fire_forecasting}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"
MODEL_SEED="${MODEL_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-6}"
DIAGNOSTIC_SAMPLES="${DIAGNOSTIC_SAMPLES:-0}"
PROJECTION="${PROJECTION:-umap}"

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_mplconfig

export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
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

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Model seed: ${MODEL_SEED}"
echo "Projection: ${PROJECTION}"
echo "Batch size: ${BATCH_SIZE}"
echo "Workers: ${NUM_WORKERS}"
echo "Diagnostic subset maximum: ${DIAGNOSTIC_SAMPLES}"
"${PYTHON_BIN}" --version
nvidia-smi

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/plot_latent_alignment_all_data.py \
  --model-seed "${MODEL_SEED}" \
  --selection-seed 42 \
  --samples-per-fire 150 \
  --projection "${PROJECTION}" \
  --projection-seed 42 \
  --diagnostic-samples "${DIAGNOSTIC_SAMPLES}" \
  --diagnostic-seed 42 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --cpu-threads 8 \
  --device cuda

echo "Completed complete-dataset FLARE latent-alignment figure."
