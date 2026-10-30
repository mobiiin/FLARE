#!/usr/bin/env bash
#SBATCH --job-name=table2_smoke
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=artifacts/logs/slurm/table2_smoke_%j.out
#SBATCH --error=artifacts/logs/slurm/table2_smoke_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mhabibp/FLARE}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"
BASELINES=("$@")
if [[ ${#BASELINES[@]} -eq 0 ]]; then
  BASELINES=(convlstm_unet earthformer_lite cawfe_st_mamba)
fi

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_table2_smoke_mplconfig
export MPLCONFIGDIR=/tmp/mhabibp_table2_smoke_mplconfig
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

echo "Table 2 GPU smoke job: ${SLURM_JOB_ID:-unknown} on ${SLURM_NODELIST:-unknown}"
nvidia-smi
"${PYTHON_BIN}" scripts/check_table2_baseline_configs.py

SMOKE_ARGS=()
for baseline in "${BASELINES[@]}"; do
  case "${baseline}" in
    convlstm_unet|earthformer_lite|cawfe_st_mamba)
      SMOKE_ARGS+=(--baseline "${baseline}")
      ;;
    *)
      echo "GPU smoke accepts learned baselines only; got ${baseline}." >&2
      exit 2
      ;;
  esac
done

echo "Running exactly two train and two validation minibatches per selected model; test is not iterated."
srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/smoke_test_table2_baselines.py "${SMOKE_ARGS[@]}"

