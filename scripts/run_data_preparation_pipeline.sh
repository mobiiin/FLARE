#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_data_preparation_pipeline.sh CONFIG [OPTIONS]

Options:
  --derived-config PATH       Derived threshold config path
  --percentile FLOAT          Threshold percentile (default: 5.0)
  --pattern PATTERN           Normalization/sample pattern (default: sparse5_h10)
  --all-patterns              Build all enabled temporal patterns (default)
  --skip-engineered           Skip engineered full-frame build
  --skip-thresholds           Skip threshold estimation
  --skip-targets              Skip target construction
  --skip-patches              Skip patch-index construction; otherwise patch index metadata is rebuilt
  --skip-samples              Skip temporal sample-index construction
  --skip-normalization        Skip normalization
  --skip-inspection           Skip inspection
  --make-quicklooks           Save debug visualizations
  --overwrite                 Pass overwrite where supported; disables engineered --skip_existing
  --max-frames-per-fire N     Debug-only frame limit
  --engineered-workers N      Concurrent full-frame builders per fire (default: 1)
  --normalization-frame-cache-gb N  RAM cache for decompressed normalization frames (default: 0)
  --dry-run                   Print commands without running them
  --help                      Show this help
EOF
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    [[ $# -ge 1 ]] && [[ "$1" == "--help" || "$1" == "-h" ]] && exit 0
    exit 1
fi

CONFIG="$1"
shift
if [[ ! -f "$CONFIG" ]]; then
    echo "Config file does not exist: $CONFIG" >&2
    exit 1
fi
CONFIG="$(realpath "$CONFIG")"
CONFIG_STEM="$(basename "$CONFIG")"
CONFIG_STEM="${CONFIG_STEM%.*}"

DERIVED_CONFIG=""
PERCENTILE="5.0"
PATTERN="sparse5_h10"
ALL_PATTERNS=1
MAKE_QUICKLOOKS=0
DRY_RUN=0
OVERWRITE=0
SKIP_ENGINEERED=0
SKIP_THRESHOLDS=0
SKIP_TARGETS=0
SKIP_PATCHES=0
SKIP_SAMPLES=0
SKIP_NORMALIZATION=0
SKIP_INSPECTION=0
MAX_FRAMES_PER_FIRE=""
ENGINEERED_WORKERS="1"
NORMALIZATION_FRAME_CACHE_GB="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --derived-config) DERIVED_CONFIG="$2"; shift 2 ;;
        --percentile) PERCENTILE="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --all-patterns) ALL_PATTERNS=1; shift ;;
        --skip-engineered) SKIP_ENGINEERED=1; shift ;;
        --skip-thresholds) SKIP_THRESHOLDS=1; shift ;;
        --skip-targets) SKIP_TARGETS=1; shift ;;
        --skip-patches) SKIP_PATCHES=1; shift ;;
        --skip-samples) SKIP_SAMPLES=1; shift ;;
        --skip-normalization) SKIP_NORMALIZATION=1; shift ;;
        --skip-inspection) SKIP_INSPECTION=1; shift ;;
        --make-quicklooks) MAKE_QUICKLOOKS=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --max-frames-per-fire) MAX_FRAMES_PER_FIRE="$2"; shift 2 ;;
        --engineered-workers) ENGINEERED_WORKERS="$2"; shift 2 ;;
        --normalization-frame-cache-gb) NORMALIZATION_FRAME_CACHE_GB="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$DERIVED_CONFIG" ]]; then
    DERIVED_CONFIG="configs/derived/${CONFIG_STEM}_with_fire_mask_thresholds.yaml"
fi
if [[ ! "$DERIVED_CONFIG" = /* ]]; then
    DERIVED_CONFIG="$(realpath -m "$DERIVED_CONFIG")"
fi

for required in \
    scripts/build_engineered_frame_dataset.py \
    scripts/estimate_fire_mask_thresholds.py \
    scripts/build_target_dataset.py \
    scripts/build_patch_index.py \
    scripts/build_temporal_sample_index.py \
    scripts/compute_processed_dataset_normalization.py \
    scripts/inspect_processed_dataset.py; do
    [[ -f "$required" ]] || { echo "Required script is missing: $required" >&2; exit 1; }
done

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="artifacts/logs/data_preparation/${CONFIG_STEM}_${TIMESTAMP}"
mkdir -p "$LOG_DIR" configs/derived
touch "$LOG_DIR/commands.txt"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

write_status() {
    local status="$1"
    local finished_at
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cat > "$LOG_DIR/status.json" <<EOF
{
  "status": "$status",
  "config": "$(printf '%s' "$CONFIG" | sed 's/"/\\"/g')",
  "derived_config": "$(printf '%s' "$DERIVED_CONFIG" | sed 's/"/\\"/g')",
  "pattern": "$PATTERN",
  "percentile": $PERCENTILE,
  "started_at": "$STARTED_AT",
  "finished_at": "$finished_at",
  "log_dir": "$(printf '%s' "$LOG_DIR" | sed 's/"/\\"/g')"
}
EOF
}
on_error() { write_status "failed"; }
trap on_error ERR

{
    echo "hostname: $(hostname)"
    echo "date: $(date -u)"
    echo "pwd: $(pwd)"
    echo "python_version: $(python --version 2>&1)"
    echo "python_path: $(command -v python || true)"
    echo "git_head: $(git rev-parse HEAD 2>/dev/null || true)"
    echo "config: $CONFIG"
    echo "derived_config: $DERIVED_CONFIG"
    echo "percentile: $PERCENTILE"
    echo "pattern: $PATTERN"
} > "$LOG_DIR/environment.txt"

run_cmd() {
    echo
    echo "============================================================"
    printf 'Running:'; printf ' %q' "$@"; echo
    echo "============================================================"
    printf '%q ' "$@" >> "$LOG_DIR/commands.txt"
    echo >> "$LOG_DIR/commands.txt"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    "$@"
}

normalization_output_dir() {
    python - "$1" <<'PY'
import sys
from pathlib import Path
from src.config import load_config

config_path = Path(sys.argv[1]).resolve()
config = load_config(str(config_path))
processed = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), dict) else {}
root = Path(processed.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser()
if not root.is_absolute():
    root = (config_path.parent / root).resolve()
normalization = config.get("normalization", {}) if isinstance(config.get("normalization"), dict) else {}
output = Path(normalization.get("output_dir", root / "normalization")).expanduser()
if not output.is_absolute():
    output = root / output
print(output.resolve())
PY
}

verify_normalization_outputs() {
    local config="$1"
    local pattern="$2"
    local pattern_alias="${pattern//\//_}"
    local norm_dir
    norm_dir="$(normalization_output_dir "$config")"
    local json_path="$norm_dir/latest_normalization_${pattern_alias}.json"
    local npz_path="$norm_dir/latest_normalization_${pattern_alias}.npz"

    echo
    echo "Verifying normalization aliases for pattern '${pattern}' in ${norm_dir}"
    if [[ ! -s "$json_path" || ! -s "$npz_path" ]]; then
        echo "ERROR: Expected normalization alias files were not created:" >&2
        echo "  $json_path" >&2
        echo "  $npz_path" >&2
        echo "Re-run manually with:" >&2
        echo "  python scripts/compute_processed_dataset_normalization.py --config $config --pattern $pattern" >&2
        exit 1
    fi
    python - "$json_path" "$pattern" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
expected_pattern = sys.argv[2]
payload = json.loads(json_path.read_text())
actual_pattern = payload.get("sample_pattern")
if actual_pattern != expected_pattern:
    raise SystemExit(
        f"Normalization alias {json_path} has sample_pattern={actual_pattern!r}, "
        f"expected {expected_pattern!r}."
    )
print(f"Verified normalization alias: {json_path}")
PY
}


verify_temporal_sample_index_fires() {
    local config="$1"
    local pattern="$2"
    python - "$config" "$pattern" <<'PY'
import json
import sys
from pathlib import Path
from src.config import load_config

config_path = Path(sys.argv[1]).resolve()
pattern_arg = sys.argv[2]
config = load_config(str(config_path))
processed = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), dict) else {}
root = Path(processed.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")).expanduser()
if not root.is_absolute():
    root = (config_path.parent / root).resolve()
manifest_path = root / "dataset_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_splits = manifest.get("splits", {}) if isinstance(manifest.get("splits"), dict) else {}
patterns = ["consecutive5_h10", "single1_h10", "sparse5_h10"] if pattern_arg == "all" else [pattern_arg]
failed = False
for pattern in patterns:
    sample_path = root / "indices" / "temporal" / f"samples_{pattern}.jsonl"
    if not sample_path.exists():
        print(f"ERROR: missing temporal sample index for pattern {pattern}: {sample_path}", file=sys.stderr)
        failed = True
        continue
    index_fires = {"train": set(), "val": set(), "test": set()}
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row.get("split"))
        if split in index_fires:
            index_fires[split].add(str(row.get("fire_name")))
    for split in ("train", "val", "test"):
        manifest_fires = manifest_splits.get(f"{split}_fires", manifest_splits.get(split, []))
        manifest_fire_set = {str(fire) for fire in manifest_fires} if isinstance(manifest_fires, list) else set()
        if manifest_fire_set and index_fires[split] != manifest_fire_set:
            print(
                f"ERROR: samples_{pattern}.jsonl {split} fires do not match dataset_manifest: "
                f"index={sorted(index_fires[split])} manifest={sorted(manifest_fire_set)}",
                file=sys.stderr,
            )
            failed = True
if failed:
    raise SystemExit(1)
print(f"Verified temporal sample indices against dataset_manifest for pattern={pattern_arg}")
PY
}

if [[ -n "$MAX_FRAMES_PER_FIRE" ]]; then
    echo "DEBUG MODE: max frames per fire is set. Dataset is incomplete."
fi

if [[ "$SKIP_ENGINEERED" == "0" ]]; then
    cmd=(python scripts/build_engineered_frame_dataset.py --config "$CONFIG")
    if [[ "$OVERWRITE" == "1" ]]; then
        cmd+=(--overwrite)
    else
        cmd+=(--skip_existing)
    fi
    cmd+=(--workers "$ENGINEERED_WORKERS")
    if [[ "$MAKE_QUICKLOOKS" == "0" ]]; then cmd+=(--no_quicklooks); fi
    [[ -n "$MAX_FRAMES_PER_FIRE" ]] && cmd+=(--max_frames_per_fire "$MAX_FRAMES_PER_FIRE")
    run_cmd "${cmd[@]}"
fi

if [[ "$SKIP_THRESHOLDS" == "0" ]]; then
    cmd=(python scripts/estimate_fire_mask_thresholds.py --config "$CONFIG" --percentile "$PERCENTILE" --update_config --derived_config_path "$DERIVED_CONFIG")
    [[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)
    run_cmd "${cmd[@]}"
else
    if [[ "$DRY_RUN" == "0" && ! -f "$DERIVED_CONFIG" ]]; then
        if ! python - "$CONFIG" <<'PY'
import sys
from src.config import load_config
from src.data.fire_mask_thresholds import resolve_frozen_thresholds
try:
    resolve_frozen_thresholds(load_config(sys.argv[1]), sys.argv[1], require=True)
except Exception:
    raise SystemExit(1)
PY
        then
            if [[ "$SKIP_TARGETS" == "0" ]]; then
                echo "Cannot build targets without thresholds. Run threshold estimation or provide --derived-config." >&2
                exit 1
            fi
            echo "WARNING: thresholds skipped and no frozen threshold config was found."
        fi
    fi
fi

TARGET_CONFIG="$DERIVED_CONFIG"
if [[ "$SKIP_THRESHOLDS" == "1" && ! -f "$TARGET_CONFIG" ]]; then TARGET_CONFIG="$CONFIG"; fi

if [[ "$SKIP_TARGETS" == "0" ]]; then
    cmd=(python scripts/build_target_dataset.py --config "$TARGET_CONFIG")
    [[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)
    run_cmd "${cmd[@]}"
fi
if [[ "$SKIP_PATCHES" == "0" ]]; then
    # Patch indices are metadata derived from dataset_manifest/fire manifests.
    # Always rebuild them on incremental dataset-prep reruns so newly added fires
    # are included even while engineered frame generation uses --skip_existing.
    cmd=(python scripts/build_patch_index.py --config "$TARGET_CONFIG" --overwrite)
    run_cmd "${cmd[@]}"
fi
if [[ "$SKIP_SAMPLES" == "0" ]]; then
    if [[ "$ALL_PATTERNS" == "1" ]]; then
        run_cmd python scripts/build_temporal_sample_index.py --config "$TARGET_CONFIG" --pattern all
        SAMPLE_PATTERN_TO_VERIFY="all"
    else
        run_cmd python scripts/build_temporal_sample_index.py --config "$TARGET_CONFIG" --pattern "$PATTERN"
        SAMPLE_PATTERN_TO_VERIFY="$PATTERN"
    fi
    if [[ "$DRY_RUN" == "0" ]]; then
        if ! verify_temporal_sample_index_fires "$TARGET_CONFIG" "$SAMPLE_PATTERN_TO_VERIFY"; then
            echo "Temporal sample indices are stale or incomplete. Rebuilding patch and temporal indices once..." >&2
            run_cmd python scripts/build_patch_index.py --config "$TARGET_CONFIG" --overwrite
            if [[ "$ALL_PATTERNS" == "1" ]]; then
                run_cmd python scripts/build_temporal_sample_index.py --config "$TARGET_CONFIG" --pattern all
            else
                run_cmd python scripts/build_temporal_sample_index.py --config "$TARGET_CONFIG" --pattern "$PATTERN"
            fi
            verify_temporal_sample_index_fires "$TARGET_CONFIG" "$SAMPLE_PATTERN_TO_VERIFY"
        fi
    fi
fi
if [[ "$SKIP_NORMALIZATION" == "0" ]]; then
    run_cmd python scripts/compute_processed_dataset_normalization.py --config "$TARGET_CONFIG" --pattern "$PATTERN" --frame_cache_gb "$NORMALIZATION_FRAME_CACHE_GB"
    if [[ "$DRY_RUN" == "0" ]]; then
        verify_normalization_outputs "$TARGET_CONFIG" "$PATTERN"
    fi
fi
if [[ "$SKIP_INSPECTION" == "0" ]]; then
    run_cmd python scripts/inspect_processed_dataset.py --config "$TARGET_CONFIG"
fi

if [[ "$MAKE_QUICKLOOKS" == "1" ]]; then
    run_cmd python scripts/visualize_engineered_frames.py --config "$TARGET_CONFIG" --split train --mode save --output_dir artifacts/data_debug/engineered_frames
    run_cmd python scripts/visualize_patch_index.py --config "$TARGET_CONFIG" --split train --mode save --output_dir artifacts/data_debug/patch_index
    run_cmd python scripts/visualize_targets.py --config "$TARGET_CONFIG" --split train --mode save --output_dir artifacts/data_debug/targets
    run_cmd python scripts/visualize_processed_samples.py --config "$TARGET_CONFIG" --pattern "$PATTERN" --split train --mode save --output_dir artifacts/data_debug/processed_samples
fi

if [[ "$DRY_RUN" == "1" ]]; then
    write_status "dry_run"
    trap - ERR
    echo "Dry run completed; no dataset files were built."
    exit 0
fi

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "success"
trap - ERR
echo
echo "Data preparation completed successfully."
echo "  Config used: $CONFIG"
echo "  Derived threshold config: $TARGET_CONFIG"
echo "  Logs: $LOG_DIR"
echo "  Next suggested inspection: python scripts/inspect_processed_dataset.py --config $TARGET_CONFIG"
echo "  Next suggested visualizations:"
echo "    python scripts/visualize_targets.py --config $TARGET_CONFIG --split train"
echo "    python scripts/visualize_processed_samples.py --config $TARGET_CONFIG --pattern $PATTERN --split train"
echo "  Next training command: python scripts/train_forecasting_model.py --config $TARGET_CONFIG"
echo "  Ablation examples:"
echo "    python scripts/train_forecasting_model.py --config configs/experiments/convlstm_consecutive5_h10.yaml"
echo "    python scripts/train_forecasting_model.py --config configs/experiments/convlstm_single1_h10.yaml"
echo "    python scripts/train_forecasting_model.py --config configs/experiments/convlstm_sparse5_h10.yaml"
