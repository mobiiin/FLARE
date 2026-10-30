#!/usr/bin/env bash
#SBATCH --job-name=flare_terrain_summary
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=artifacts/logs/slurm/flare_terrain_summary_%j.out
#SBATCH --error=artifacts/logs/slurm/flare_terrain_summary_%j.err
set -euo pipefail
cd /home/mhabibp/FLARE
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fire_forecasting
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/summarize_terrain_ablation.py
