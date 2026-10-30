#!/usr/bin/env bash
#SBATCH --job-name=flare_data_prep
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=/home/mhabibp/FLARE/artifacts/logs/slurm/flare_data_prep_%j.out
#SBATCH --error=/home/mhabibp/FLARE/artifacts/logs/slurm/flare_data_prep_%j.err

set -euo pipefail

cd /home/mhabibp/FLARE
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fire_forecasting
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Rebuilding the complete CAWFE processed dataset on scratch (CPU only)."
echo "Raw index: fire_dataset_index_trimmed.json"
echo "Output: /scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1"
echo "This repeats the prior full preparation settings: percentile 5, all temporal patterns,"
echo "eight engineered-frame workers, and a 96 GiB normalization frame cache."

srun --ntasks=1 bash scripts/run_data_preparation_pipeline.sh configs/default.yaml \
  --percentile 5.0 \
  --pattern sparse5_h10 \
  --all-patterns \
  --engineered-workers 8 \
  --normalization-frame-cache-gb 96

echo "Preparing the fixed screening-validation subset used by FLARE training."
srun --ntasks=1 python scripts/prepare_cawfe_latte_screening_validation.py \
  --config configs/ablations/cawfe_latte_baseline.yaml

echo "Processed dataset rebuild, inspection, and FLARE validation preparation completed."
