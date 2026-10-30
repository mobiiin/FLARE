#!/usr/bin/env bash
#SBATCH --job-name=table2_summary
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=artifacts/logs/slurm/table2_summary_%j.out
#SBATCH --error=artifacts/logs/slurm/table2_summary_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mhabibp/FLARE}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"
cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_table2_summary_mplconfig
export MPLCONFIGDIR=/tmp/mhabibp_table2_summary_mplconfig
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python after activating ${CONDA_ENV}." >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Table 2 aggregation job: ${SLURM_JOB_ID:-unknown} on ${SLURM_NODELIST:-unknown}"
nvidia-smi
"${PYTHON_BIN}" scripts/check_table2_baseline_configs.py
srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/summarize_table2_results.py "$@"

