#!/usr/bin/env bash
set -euo pipefail
VARIANTS=(TERRAIN_NONE TERRAIN_EARLY_CONCAT TERRAIN_PREFUSION_FILM TERRAIN_QUERY TERRAIN_LATE_CONCAT TERRAIN_POSTFUSION_FILM)
SEEDS=(42 123 2026)
DRY_RUN=false
JOB_IDS=()
usage() { echo "Usage: bash $0 [--seeds 42 123 2026] [--variants TERRAIN_QUERY ...] [--dry-run]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      shift; SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
      [[ ${#SEEDS[@]} -gt 0 ]] || { usage >&2; exit 2; }
      ;;
    --variants)
      shift; VARIANTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do VARIANTS+=("$1"); shift; done
      [[ ${#VARIANTS[@]} -gt 0 ]] || { usage >&2; exit 2; }
      ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
cd /home/mhabibp/FLARE
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python scripts/run_terrain_ablation.py "$variant" "$seed" --check-only
  done
done
if [[ "$DRY_RUN" == true ]]; then
  for variant in "${VARIANTS[@]}"; do for seed in "${SEEDS[@]}"; do
    echo "sbatch scripts/slurm_terrain_ablation_a10080.sh $variant $seed"
  done; done
  exit 0
fi
mkdir -p artifacts/logs/slurm
for variant in "${VARIANTS[@]}"; do for seed in "${SEEDS[@]}"; do
  run_dir="artifacts/terrain_integration_ablation/${variant}/seed_${seed}"
  if [[ -f "${run_dir}/terrain_ablation_metadata.json" ]]; then
    echo "${variant} seed=${seed}: complete run exists; skipping"
    continue
  fi
  recover_arg=()
  if [[ -e "${run_dir}" ]]; then
    if [[ ! -f "${run_dir}/checkpoints/latest_model.pt" ]]; then
      echo "${variant} seed=${seed}: incomplete run lacks latest checkpoint: ${run_dir}" >&2
      exit 1
    fi
    recover_arg=(--recover)
  fi
  job_id="$(sbatch --parsable scripts/slurm_terrain_ablation_a10080.sh "$variant" "$seed" "${recover_arg[@]}")"
  JOB_IDS+=("${job_id}")
  echo "${variant} seed=${seed}: submitted job ${job_id} ${recover_arg[*]}"
done; done
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
  dependency="$(IFS=:; echo "${JOB_IDS[*]}")"
  summary_id="$(sbatch --parsable --dependency="afterany:${dependency}" scripts/slurm_summarize_terrain_ablation.sh)"
  echo "Validation summary: submitted dependent job ${summary_id}"
else
  python scripts/summarize_terrain_ablation.py
fi
