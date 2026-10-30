#!/usr/bin/env bash
#SBATCH --job-name=flare_terrain
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=72:00:00
#SBATCH --chdir=/home/mhabibp/FLARE
#SBATCH --output=artifacts/logs/slurm/flare_terrain_%j.out
#SBATCH --error=artifacts/logs/slurm/flare_terrain_%j.err
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then echo "Usage: sbatch $0 TERRAIN_VARIANT SEED [--recover]" >&2; exit 2; fi
VARIANT="$1"
SEED="$2"
RECOVER_FLAG="${3:-}"
if [[ -n "$RECOVER_FLAG" && "$RECOVER_FLAG" != --recover ]]; then echo "Third argument must be --recover" >&2; exit 2; fi
cd /home/mhabibp/FLARE
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fire_forecasting
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export FIRE_FORECASTING_PROGRESS_BAR=0
python - "$VARIANT" "$SEED" <<'PY'
import sys, torch
if not torch.cuda.is_available():
    raise SystemExit('Terrain ablation requires an A100 GPU; CUDA is unavailable.')
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
if 'A100' not in name or major < 8:
    raise SystemExit(f'Terrain ablation requires A100; allocated {name} sm_{major}{minor}. The current PyTorch build cannot run on P100.')
print(f'GPU: {name} sm_{major}{minor}; variant={sys.argv[1]}; seed={sys.argv[2]}')
PY
python scripts/run_terrain_ablation.py "$VARIANT" "$SEED" --check-only
srun --ntasks=1 python scripts/run_terrain_ablation.py "$VARIANT" "$SEED" ${RECOVER_FLAG:+"$RECOVER_FLAG"}
