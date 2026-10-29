# FLARE: Wildfire Forecasting

## Overview
FLARE forecasts future wildfire state from historical simulation frames. The repository retains `cawfe_latte` architecture identifiers and checkpoint paths for compatibility with frozen experiments; paper figures use FLARE baseline and FLARE final. ConvLSTM U-Net remains a comparison model.

The current default setup is still the multitask forecast introduced earlier:
- output channel `0`: surface consumed fuel
- output channel `1`: canopy consumed fuel
- output channel `2`: active fire / perimeter mask logits
- output channel `3`: `log1p` energy release total MW

## Qualitative test figure

To compare ground truth, ConvLSTM U-Net, FLARE baseline, and FLARE final for one fire, run:

```bash
python scripts/qualitative_test_comparison.py CHIMNEYTOPS2 10
```

The arguments are a fire name or `test`, `val`, `train`, or `all`, followed by the number of PNGs per fire. The command picks different random timestamps and writes PNGs under `artifacts/qualitative_test_comparison/<split>/<fire>/`. See [the qualitative figure guide](research/qualitative_test_comparison.md) for details.

For the held-out-test paper figure showing current, ground-truth future, and predicted future **burn-footprint boundaries** for FLARE baseline and FLARE final, see [the boundary figure guide](research/fire_front_displacement.md). It uses a GT-only frozen case selection and the Table 2 checkpoints.

## FLARE failure-case PNGs

Generate ten figures from objectively ranked RIM and Thomas held-out test failures with one command:

```bash
python scripts/failure_cases.py 10
```

The number is the **total** PNG count. The default selection covers composite, Dice, HRR Log MAE, and Active Canopy MAE failures in both fires. It includes the worst distinct case for each fire and criterion, then samples additional cases from each top-50 pool with a fixed selection seed. Only PNGs are written to a run-specific folder under `artifacts/failure_cases/png_gallery/`. Optional flags include `--fire rim`, `--fire thomas`, `--criterion hrr`, `--top-k 50`, `--selection-seed 42`, and `--model-seed 42`.

## Conda Environment Setup
Create and activate the environment before running any scripts:

```bash
conda create -n fire_forecasting python=3.10 -y
conda activate fire_forecasting
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Format
Each timestamp file is expected to have shape:

```text
(144, 144, 86)
```

Default channel layout:
- `0-79`: atmospheric variables
- `80-85`: flux + fuel variables

Configured via:

```yaml
channel_layout:
  atmospheric_channels: [0, 79]
  flux_channels: [80, 81, 82, 83]
  fuel_channels: [84, 85]
  surface_fuel_channel: 84
  canopy_fuel_channel: 85
  flux_mask_channel: 80
```

## Dataset Splitting Policy
## Training On Multiple Simulation Datasets
Use `data_dirs` to provide multiple dataset folders. Each folder is treated as an independent time series, so input sequences never cross folder boundaries.

In `multi_dataset_chronological` mode:
- each dataset is split chronologically into train/validation/test independently
- train splits are concatenated across datasets
- validation splits are concatenated across datasets
- test splits are concatenated across datasets
- no random frame splitting is used

Normalization rules:
- normalization stats are computed only from the combined training splits
- validation and test splits are not used for normalization
- sequences, previous-frame deltas, and initial-fuel references all stay local to the source dataset

This improves generalization by exposing the model to more fire regimes while still preserving chronological structure inside each simulation.

Example config:

```yaml
data_dirs:
  - ../keepz_08
  - ../keepz_08_tuo
split_mode: multi_dataset_chronological
train_fraction: 0.7
val_fraction: 0.15
test_fraction: 0.15
```

If the goal is strict cross-dataset generalization, add a future split mode that holds out whole datasets. The current `multi_dataset_chronological` mode uses all configured datasets in train/val/test splits chronologically.

## Inputs And Targets
Base input channels: `86`

With the default engineered features enabled, the dataset appends:
- `4` flux delta channels
- `2` fuel delta channels
- `2` step consumed fuel channels
- `2` cumulative consumed fuel channels
- `8` horizontal wind speed channels
- `1` low-level mean wind speed channel
- `8` updraft channels
- `16` wind direction unit-vector channels

Total input channels:

```text
86 + 10 + 33 = 129
```

Default model config:

```yaml
model:
  input_channels: 129
  output_channels: 4
```

## Energy Release Map Target
CAWFE provides sensible and latent heat fluxes in `W/m^2`. This project derives an additional downstream target called the energy release map by multiplying the summed fluxes by grid-cell area from `dx * dy`.

Formula:

```text
P_energy_MW(x,y,t) =
dx * dy *
(Q_sens_surface + Q_sens_canopy + Q_lat_surface + Q_lat_canopy)
/ 1e6
```

This is intentionally called the energy release map, not FRP, because the CAWFE inputs here are sensible and latent heat fluxes rather than radiative power. The model predicts `log1p(P_energy_MW)` as an additional continuous output target because the distribution is sparse and heavy-tailed. Evaluation reports both log-space error and physical MW error.

Per-cell area now comes from each fire dataset's `.geom` file rather than a single scalar `dx * dy`. The recommended discovery/index workflow is:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
```

The resulting `fire_dataset_index.json` is written to the project root by default and is used to pair processed `.npy` folders with the matching geometry source. If energy release is enabled, old 3-channel checkpoints are incompatible with the 4-channel model head and must not be reused without retraining.

## Manual Fire Splits
For cross-fire generalization experiments, the project now supports assigning whole fires to train, validation, and test with `split_mode: manual_fire_holdout` in [configs/default.yaml]. In this mode, `manual_fire_split.train_fires`, `val_fires`, and `test_fires` must match keys in `fire_dataset_index.json`, and validation/test fires are completely held out from training and normalization.

Recommended workflow:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
python scripts/precompute_patch_cache.py --config configs/default.yaml --split train
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/precompute_patch_cache.py --config configs/default.yaml --split val
python scripts/precompute_patch_cache.py --config configs/default.yaml --split test
python scripts/inspect_patch_cache.py --config configs/default.yaml --split all
python scripts/train_forecasting_model.py --config configs/default.yaml
```

Choose training fires to cover a range of fire sizes, durations, energy-release regimes, and spatial extents. Hold out validation/test fires as unseen events when you want a scientifically cleaner cross-fire generalization result.

## Manual Fire Start Trimming
Use the manual trim workflow after dataset discovery when you want to remove inactive pre-fire frames. The raw CAWFE `.npy` files are not modified. The trimmed index stores compact per-fire `temporal_trim` start/end indices only, so it does not duplicate thousands of frame paths.

```bash
python scripts/manual_trim_fire_datasets.py \
  --input_index fire_dataset_index.json \
  --output_index fire_dataset_index_trimmed.json
```

To review only one fire:

```bash
python scripts/manual_trim_fire_datasets.py \
  --input_index fire_dataset_index.json \
  --only_fire MCKINNEY \
  --output_index fire_dataset_index_trimmed.json
```

To apply existing choices without opening the interactive reviewer:

```bash
python scripts/manual_trim_fire_datasets.py \
  --apply_only \
  --input_index fire_dataset_index.json \
  --trim_config configs/manual_fire_trim.json \
  --output_index fire_dataset_index_trimmed.json
```

Inspect the result before training:

```bash
python scripts/inspect_multi_fire_dataset.py \
  --index fire_dataset_index_trimmed.json \
  --config configs/default.yaml
```

Then point `fire_dataset_index_json` in `configs/default.yaml` at `../fire_dataset_index_trimmed.json` when you are ready to use it. Dataset loading and patch-cache precompute interpret local sample starts through each fire's `temporal_trim` metadata, preserving original frame indices in sample metadata. After trimming frames, rebuild the patch cache and normalization stats because old cached patches/statistics are incompatible with the new frame sequences. Bump `cache.cache_version` when switching trim decisions.

```bash
python scripts/precompute_patch_cache.py --config configs/default.yaml --split all
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
```

## Fast Patch Cache Training
The default config can train from precomputed patch shards under `/scratch/mhabibp/fire_forecasting_patch_cache/`. This avoids reopening full `.npy` frames and rebuilding full-domain engineered features, multitask targets, and energy-release maps inside every training batch.

Use `scripts/precompute_patch_cache.py` to save `X` patches shaped `(N,T,C,H,W)` and `y` patches shaped `(N,4,H,W)`. Inputs are stored unnormalized by default, so compute normalization from the train cache before training:

```bash
python scripts/precompute_patch_cache.py --config configs/default.yaml --split train
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/precompute_patch_cache.py --config configs/default.yaml --split val
python scripts/precompute_patch_cache.py --config configs/default.yaml --split test
python scripts/inspect_patch_cache.py --config configs/default.yaml --split all
python scripts/train_forecasting_model.py --config configs/default.yaml
```

If `/scratch` purges the cache, training raises a clear error and asks you to rerun precompute unless `cache.allow_dynamic_fallback: true` is set.

## Visualizing Patch Cache Samples
Use the standalone patch-cache viewer to inspect cached `X`/`y` samples directly from scratch without modifying the cache:

```bash
python scripts/visualize_patch_cache.py \
  --config configs/default.yaml \
  --split train

python scripts/visualize_patch_cache.py \
  --config configs/default.yaml \
  --split val \
  --sample_index 100

python scripts/visualize_patch_cache.py \
  --config configs/default.yaml \
  --split test \
  --mode save \
  --random \
  --num_save_samples 10
```

Useful controls: `n`/`p` for next/previous sample, up/down for input frame, `g`/`G` for channel group, `v` to cycle views, `w` to save the current figure, and `q` to quit.

## Experiment Configs
Every launchable training/evaluation script should be run with an explicit config:

```bash
python scripts/precompute_patch_cache.py --config configs/experiments/convlstm_debug_fixed_subset.yaml --split all
python scripts/compute_normalization.py --config configs/experiments/convlstm_debug_fixed_subset.yaml --from_cache
python scripts/train_forecasting_model.py --config configs/experiments/convlstm_debug_fixed_subset.yaml
```

Experiment configs can inherit from `configs/default.yaml` with `base_config` and override only what changes. The loader recursively merges the config and resolves simple `${paths.*}` references, so each experiment can own separate patch-cache, normalization, run, log, and result directories:

```yaml
base_config: ../default.yaml
experiment:
  name: convlstm_debug_fixed_subset
paths:
  patch_cache_root: ${paths.scratch_root}/fire_forecasting_patch_cache/convlstm_debug_fixed_subset
  artifacts_root: artifacts/experiments/convlstm_debug_fixed_subset
```

Training runs save the exact submitted config as `configs/original_config.yaml`, the fully resolved config as `configs/resolved_config.yaml` and `configs/config_used.yaml`, plus hashes and inheritance metadata in `metadata/config_metadata.json`. Patch-cache manifests and normalization sidecars also record config and dataset-index hashes, which makes parallel Slurm experiments reproducible and easier to audit.

Generic Palmetto wrappers:

```bash
sbatch scripts/slurm_precompute_cache_with_config.sh configs/experiments/convlstm_debug_fixed_subset.yaml all
sbatch scripts/slurm_train_with_config_a10080.sh configs/experiments/convlstm_debug_fixed_subset.yaml
```

## Input Normalization Consistency
Input normalization is fit once from train samples only. The same saved train stats are then reused for train, validation, test, debug, and paper evaluation. With the default `input_normalization_device: cuda`, datasets return raw inputs and the shared training/evaluation code applies z-score normalization on the model device immediately before `model(x)`.

Each normalization run writes timestamped, config-aware files under `normalization.output_dir`. The default points that directory at `paths.patch_cache_root`, so normalization stats sit beside the scratch patch cache:

```text
train_normalization_stats_<config_name>_<YYYYMMDD_HHMMSS>.json
train_normalization_stats_<config_name>_<YYYYMMDD_HHMMSS>.npz
normalization_stats.json
normalization_stats.npz
```

For this project, configs point the stable aliases directly into the patch-cache directory:

```yaml
normalization:
  output_dir: ${paths.patch_cache_root}
  stats_path: ${paths.patch_cache_root}/normalization_stats.json
  npz_path: ${paths.patch_cache_root}/normalization_stats.npz
```

The JSON metadata records the exact timestamped NPZ path, config name/path/hash, cache directory/version, dataset-index hash, fit split, and channel count. Training copies the JSON metadata into the run directory and records the exact NPZ path/hash without copying the potentially large NPZ unless `training.output.copy_normalization_npz: true`.

Useful checks:

```bash
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
python scripts/inspect_normalization_pipeline.py --config configs/default.yaml --split all
python scripts/sanity_check_project.py --config configs/default.yaml
```

New checkpoints record normalization metadata in the checkpoint and run summary. Evaluation checks that metadata when present; use `--allow_normalization_mismatch` only for intentional compatibility/debug runs with older or deliberately different stats.

## Training Performance Optimization
All model-specific training scripts now route through the shared optimized training path in `src/training/train.py`. The shared path supports GPU-aware batch sizing, BF16/FP16 AMP, TF32, GPU-side input normalization, shard-local cache batching, Slurm-aware DataLoader worker caps, timing CSV logs, optional CUDA prefetching, and optional `torch.compile`.

Useful commands:

```bash
python scripts/diagnose_training_pipeline.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --num_batches 50

python scripts/benchmark_patch_cache_io.py \
  --config configs/default.yaml \
  --split train \
  --num_batches 100

python scripts/train_forecasting_model.py \
  --config configs/default.yaml
```

On Palmetto, request enough CPUs for the DataLoader, keep the patch cache on `/scratch/mhabibp/`, and prefer BF16 on A100/H100 GPUs. If GPU utilization is low and timing logs show high `data_wait`, tune workers or cache locality first. If VRAM is underused and `data_wait` is low, increase batch size or enable `training.performance.auto_batch_size`.

## Training Run Outputs
Every training entry point now creates a unique run directory:

```text
artifacts/runs/<architecture>/<run_name>/
```

The run directory is the source of truth for checkpoints, logs, configs, metadata, and training curves. For example, the best ConvLSTM checkpoint for a run is:

```text
artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt
```

Each run stores `checkpoints/best_model.pt`, `checkpoints/latest_model.pt`, `logs/training_log.csv`, `logs/validation_log.csv`, `logs/timing_log.csv`, `figures/loss_curves.png`, `figures/metric_curves.png`, `configs/resolved_config.yaml`, and `metadata/run_summary.json`. Training no longer writes duplicate compatibility checkpoint copies under `artifacts/checkpoints/<architecture>/`; use the checkpoint inside the specific run directory.

Useful commands:

```bash
python scripts/list_training_runs.py --root artifacts/runs

python scripts/verify_training_outputs.py --all --root artifacts/runs

python scripts/plot_training_curves.py \
  --run_dir artifacts/runs/convlstm_unet/<run_name>
```

You can also name a run explicitly:

```bash
python scripts/train_forecasting_model.py \
  --config configs/default.yaml \
  --run_name convlstm_unet_main_seed42
```

The loss and metric curves are debugging plots for overfitting, instability, and training-speed inspection; they are not paper figures.

## Validation Protocol
Training uses one validation policy for the whole run. Mixing capped validation on most epochs with full validation every few epochs can make loss, Dice, IoU, active canopy consumed MAE, and active energy log MAE oscillate because the epochs are not evaluated on the same data.

The default is a fixed validation subset:

```yaml
training:
  validation:
    mode: fixed_subset_every_epoch
    max_val_batches_per_epoch: 50
    fixed_subset_seed: 42
```

In this mode, the training loop selects one deterministic set of validation batches at the beginning of training and reuses that exact subset every epoch. This keeps curves comparable while avoiding the cost of full validation.

For final model training, use full validation every epoch:

```yaml
training:
  validation:
    mode: full_every_epoch
    max_val_batches_per_epoch: null
```

After training, run full held-out evaluation with:

```bash
python scripts/evaluate_trained_models.py --config configs/default.yaml
```

## Early Stopping
All architectures that use the shared training loop support configurable early stopping. The default monitors `val_loss` on the same validation protocol used for checkpoint selection.

```yaml
training:
  max_epochs: 50
  early_stopping:
    enabled: true
    monitor: val_loss
    mode: min
    patience: 8
    min_delta: 0.001
    start_epoch: 5
    save_latest_on_stop: true
    stop_on_nan: true
```

CLI overrides:

```bash
python scripts/train_forecasting_model.py \
  --config configs/default.yaml \
  --early_stopping_patience 12 \
  --early_stopping_min_delta 0.0005
```

Use `--disable_early_stopping` for a full `max_epochs` run. Stop state is saved in checkpoints, `logs/training_log.csv`, and `metadata/run_summary.json`.

## Evaluating Trained Models
Use the paper-results evaluator after training jobs have produced run directories under `artifacts/runs/<architecture>/<run_name>/`. The script searches the requested architecture's run folders, selects the best checkpoint by saved validation-selection metadata, evaluates the held-out split with the current horizon/config, and writes a compact Markdown report plus optional JSON/CSV/TEX outputs.

Evaluate ConvLSTM on the test split using the auto-selected best checkpoint:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture convlstm_unet \
  --paper_energy_metric log
```

The fourth output/target channel remains `log1p(energy_release_MW)`. The paper table reports `Energy Log MAE` and `Active Energy Log MAE` by default, computed directly on that channel; MW-space energy metrics are still saved as secondary diagnostics because `expm1` can amplify log-space outliers.

Evaluate ConvLSTM using an explicit checkpoint:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --paper_energy_metric log
```

Evaluate baselines only:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture baseline \
  --paper_energy_metric log
```

Evaluate all learned models and baselines:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode quantitative \
  --split test \
  --model_architecture all \
  --paper_energy_metric log
```

Run quantitative and qualitative evaluation in one command:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode all \
  --split test \
  --model_architecture all \
  --num_samples 10 \
  --paper_energy_metric log
```

Primary outputs are written to:

```text
artifacts/results/<eval_name>/evaluation_report.md
```

JSON, CSV, and LaTeX sidecars are opt-in. Add `--include_json_outputs`, `--include_csv_outputs`, or `--write_latex` when you need those files.

Generate qualitative figures for a single learned checkpoint:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode qualitative \
  --split test \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --num_samples 10 \
  --qualitative_seed 42
```

Generate qualitative figures for all available learned models plus baselines:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/default.yaml \
  --mode qualitative \
  --split test \
  --model_architecture all \
  --num_samples 10 \
  --qualitative_seed 42
```

Qualitative mode uses the same selected sample indices for every evaluated model. It writes one summary image per sample by default, plus reports and selection metadata:

```text
artifacts/results/qualitative/<eval_name>/images/sample_000.png
artifacts/results/qualitative/<eval_name>/qualitative_report.md
artifacts/results/qualitative/<eval_name>/qualitative_report.json
artifacts/results/qualitative/<eval_name>/selected_samples.json
artifacts/results/qualitative/<eval_name>/selected_models.json
```

Use `--qualitative_output_format pdf` for PDF figures, `--qualitative_dpi` to change image resolution, and `--save_individual_model_panels` only when you also want per-model single-panel files. Qualitative mode is one-shot; it does not perform recursive rollout.

## Debugging Model Predictions
Use the prediction debug utility when a checkpoint looks suspicious, for example when a model performs worse than persistence or appears to overpredict energy/background activity.

```bash
python scripts/debug_model_predictions.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --split test \
  --num_batches 2
```

The script loads the checkpoint through the shared model factory and dataloader path, applies the same input-normalization path used during training/evaluation, prints target and prediction channel statistics, computes quick metrics, and saves prediction figures. It is meant to catch wrong checkpoints, output scale mismatches, energy explosions, mask collapse, channel-order mistakes, and unconstrained fuel/energy outputs.

### ConvLSTM Mask-Gated Regression
ConvLSTM U-Net can optionally gate its continuous regression outputs with the predicted fire probability to reduce diffuse positive predictions in inactive background. Channels 0, 1, and 3 are activated then multiplied by `sigmoid(mask_logits)`; channel 2 remains raw mask logits. A background suppression loss can also penalize nonzero surface/canopy/energy predictions and high mask probability in inactive target regions.

```yaml
convlstm_unet:
  use_mask_gated_regression: true
  regression_activation: relu
  mask_gate_mode: soft
  detach_mask_gate: false
  mask_gate_min: 0.0
  output_bias_init:
    enabled: true
    regression_bias: -2.0
    mask_bias: -2.0
    energy_bias: -2.0

training:
  loss:
    background_suppression:
      enabled: true
      architectures: [convlstm_unet]
      weight: 0.05
```

Train normally with:

```bash
python scripts/train_forecasting_model.py \
  --config configs/default.yaml
```

### ConvLSTM Diagnostic Evaluation
For ConvLSTM checkpoints, the same debug utility can run diagnostic-only analyses without retraining or changing model weights:

```bash
python scripts/debug_model_predictions.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --checkpoint artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
  --split test \
  --num_batches 20 \
  --run_background_diagnostics \
  --run_mask_gating_diagnostics \
  --run_oracle_gating_diagnostics
```

These diagnostics report inactive-vs-active prediction statistics, predicted-mask gated metrics, oracle target-mask gated metrics, and output activation inspection. Gating is diagnostic only: channels 0, 1, and 3 are multiplied after prediction, channel 2 remains unchanged, and the saved checkpoint/model is not modified.

Compare two checkpoints on the same selected batches:

```bash
python scripts/debug_model_predictions.py \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --split test \
  --num_batches 20 \
  --compare_checkpoints \
    artifacts/runs/convlstm_unet/<run_name>/checkpoints/best_model.pt \
    artifacts/runs/convlstm_unet/<run_name>/checkpoints/latest_model.pt \
  --checkpoint_labels best latest
```

Outputs are written under:

```text
artifacts/debug_predictions/<model_architecture>/<timestamp>/
```

## Atmospheric Engineered Features
From the raw atmospheric `U`, `V`, and `W` channels, the dataset can append three atmospheric feature groups for every input timestep.

1. Horizontal wind speed for each retained z-level:

```text
sqrt(U^2 + V^2)
```

2. Low-level mean wind speed using `low_level_indices`, default `[0, 1, 2]`:

```text
sqrt(mean(U_low)^2 + mean(V_low)^2)
```

3. Updraft for each retained z-level:

```text
max(W, 0)
```

4. Wind direction unit-vector features for each retained z-level:

```text
wind_speed = sqrt(U^2 + V^2 + eps)
wind_dir_cos = U / wind_speed
wind_dir_sin = V / wind_speed
```

These encode the direction the wind is blowing toward.

Why these features help:
- horizontal wind speed directly represents wind magnitude
- low-level wind is strongly related to near-surface fire spread
- updraft captures plume and convection strength
- wind-direction unit vectors preserve directional information without angle wraparound

Why not raw angles:
- raw `atan2`-style angle channels wrap at `360/0` degrees, which creates an artificial discontinuity for ML
- normalized `cos/sin` direction components are smoother and easier for the model to learn from
- meteorological wind direction is usually the direction wind comes from, but these features use the toward-direction vector because it is more directly aligned with transport and spread

Default channel-count example:
- raw channels: `86`
- fuel/flux engineered channels: `10`
- atmospheric engineered channels:
  `8` horizontal wind speed
  `1` low-level mean wind speed
  `8` updraft
  `16` wind direction unit-vector channels
  total atmospheric engineered = `33`
- total model input channels: `129`

## Leakage Rule
- input features may use time `t` and earlier only
- labels may use `t_future = t + prediction_horizon`
- engineered deltas and consumed-fuel input features must never use `t_future` or later frames

Why fuel/flux history is allowed as input:
- past fuel/flux are known historical state
- future fuel/perimeter are prediction targets

## Default Forecast Target
The default paper task uses `input_sequence_length: 5` and `prediction_horizon: 10`.

For sample start index `s`:
- input frame indices are `[s, s+1, s+2, s+3, s+4]`
- last input index is `s + 4`
- target index is `s + 4 + 10 = s + 14`
- valid sample count per fire is `num_frames - input_sequence_length - prediction_horizon + 1`

The multitask output remains `(B, 4, H, W)`: surface consumed fuel and canopy consumed fuel are computed from the last input frame to the horizon target frame, while the mask and `log1p` energy map come from the target frame. Rebuild the patch cache and recompute normalization stats after changing these temporal settings.

## Earthformer-lite Transformer Baseline
This project now includes `earthformer_lite`, an in-project simplified Earthformer-inspired model for comparison. It is not the full official Earthformer implementation.

The design uses axial space-time attention inspired by cuboid attention:
- temporal attention
- height attention
- width attention

Canonical patch-cache input/output:
- input: `(B, 5, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

The model predicts:
- surface consumed fuel
- canopy consumed fuel
- mask logits
- `log1p` energy release

Commands:

Smoke test:
```bash
python scripts/smoke_test_earthformer_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_earthformer_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_earthformer_lite.py --config configs/default.yaml --checkpoint artifacts/runs/earthformer_lite/<run_name>/checkpoints/best_model.pt --split test
```

Compare baselines and model:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/runs/earthformer_lite/<run_name>/checkpoints/best_model.pt --model_architecture earthformer_lite
```

## ST-Mamba-Lite Architecture
This project also includes `st_mamba_lite`, a CAWFE-tailored spatiotemporal Mamba model for dense wildfire forecasting. It is inspired by MetMamba's route-based 3D scanning ideas and ST-Mamba's ST-Mixer / ST-SSM design pattern, but it is not an official reproduction of either paper.

Canonical patch-cache input/output:
- input: `(B, 5, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

Output channels:
- surface consumed fuel
- canopy consumed fuel
- fire/perimeter mask logits
- `log1p` energy release map

The model uses:
- a per-timestep CNN stem
- route-based spatial-temporal Mamba scans over `(T, H, W)`
- local depthwise 3D convolution mixing
- temporal readout to a final 2D map
- a U-Net-style decoder for dense output reconstruction

Commands:

Smoke test:
```bash
python scripts/smoke_test_st_mamba_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_st_mamba_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_st_mamba_lite.py --config configs/default.yaml --checkpoint artifacts/runs/st_mamba_lite/<run_name>/checkpoints/best_model.pt --split test
```

Compare with baselines:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/runs/st_mamba_lite/<run_name>/checkpoints/best_model.pt --model_architecture st_mamba_lite
```

For real training, install `mamba-ssm` and set `st_mamba_lite.mamba_backend: mamba_ssm`. If `mamba_backend: auto` and `mamba-ssm` is unavailable, the code falls back to a gated sequence block meant for smoke testing and debugging only. Do not treat fallback-backend results as a publishable Mamba comparison.

OOM troubleshooting:
- reduce `batch_size`
- reduce `earthformer_lite.embed_dim`
- reduce `earthformer_lite.depths`
- reduce `earthformer_lite.num_heads`
- increase `training.gradient_accumulation_steps`
- keep `training.mixed_precision: true`

## WeatherFormer-lite Architecture
This project also includes `weatherformer_lite`, an in-project WeatherFormer-inspired factorized transformer tailored to CAWFE wildfire forecasting. It is not the official WeatherFormer implementation.

Canonical patch-cache input/output:
- input: `(B, 5, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

Output channels:
- surface consumed fuel
- canopy consumed fuel
- fire/perimeter mask logits
- `log1p` energy release map

The model uses:
- learnable channel scaling and feature gating over the CAWFE input channels
- a per-timestep CNN stem
- temporal attention at each spatial cell
- local spatial window attention with optional cyclic shifted windows
- temporal readout to final 2D maps
- a U-Net-style decoder for dense prediction

Commands:

Smoke test:
```bash
python scripts/smoke_test_weatherformer_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_weatherformer_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_weatherformer_lite.py --config configs/default.yaml --checkpoint artifacts/runs/weatherformer_lite/<run_name>/checkpoints/best_model.pt --split test
```

Compare with baselines:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/runs/weatherformer_lite/<run_name>/checkpoints/best_model.pt --model_architecture weatherformer_lite
```

Notes:
- this is a CAWFE-specific WeatherFormer-lite adaptation, not an official reproduction
- the shifted-window path currently uses cyclic shifts without a masking scheme
- channel `2` remains logits; the model does not apply sigmoid internally

## Sliding-Window Patch Cache
The project now supports deterministic sliding-window patchification for train, validation, and test. The current default is:
- `patch_size = 64`
- `stride = 60`

This gives nearly non-overlapping patches with a 4-pixel overlap while still covering the full fire domain, including border-aligned patches. Compared with the earlier sampled-train / sliding-eval behavior, it keeps training and evaluation patchification consistent.

Commands:

Rebuild the sliding-window patch cache:
```bash
python scripts/precompute_patch_cache.py --config configs/default.yaml --split all
```

Inspect the cache:
```bash
python scripts/inspect_patch_cache.py --config configs/default.yaml
```

Compute normalization from the rebuilt cache:
```bash
python scripts/compute_normalization.py --config configs/default.yaml --from_cache
```

Train:
```bash
python scripts/train_forecasting_model.py --config configs/default.yaml
```

Notes:
- train, validation, and test now all use sliding-window patchification
- the train cache is much larger than the older sampled-train cache, so one epoch can be substantially longer
- if needed, reduce epochs or set `training.max_train_batches_per_epoch`
- after changing patch stride, patch mode, input sequence length, prediction horizon, or target definition, rebuild the scratch cache or bump `cache.cache_version`
- old sampled or stride-32 caches are incompatible with the current config

## ST-Mamba-Lite Baseline
This project is also being extended with `st_mamba_lite`, a CAWFE-tailored spatial-temporal Mamba baseline for dense wildfire forecasting. It should not be described as an official reproduction of any prior Mamba paper.

Planned/expected patch-cache input/output:
- input: `(B, 5, 129, 64, 64)`
- output: `(B, 4, 64, 64)`

Planned prediction targets:
- surface consumed fuel
- canopy consumed fuel
- mask logits
- `log1p` energy release

Expected command syntax for the model-specific scripts:

Smoke test:
```bash
python scripts/smoke_test_st_mamba_lite.py --config configs/default.yaml
```

Train:
```bash
python scripts/train_st_mamba_lite.py --config configs/default.yaml
```

Test:
```bash
python scripts/test_st_mamba_lite.py --config configs/default.yaml --checkpoint artifacts/runs/st_mamba_lite/<run_name>/checkpoints/best_model.pt --split test
```

Expected comparison command once the architecture scripts are present:
```bash
python scripts/evaluate_all_baselines.py --config configs/default.yaml --split test --include_model --checkpoint artifacts/runs/st_mamba_lite/<run_name>/checkpoints/best_model.pt --model_architecture st_mamba_lite
```

## Multitask Labels
For sample start index `i`:
- input frames: `i ... i + input_sequence_length - 1`
- current time: `t = i + input_sequence_length - 1`
- future time: `t_future = t + prediction_horizon`

Label definitions:

```text
surface consumed fuel = current surface fuel - future surface fuel
canopy consumed fuel = current canopy fuel - future canopy fuel
```

Mask target modes:

`active_flux`:

```text
mask = future flux channel > flux_fire_threshold
```

`burned_fuel`:

```text
mask = max(initial fuel - future surface/canopy fuel) > consumed_fuel_threshold
```

## Reconstructing Future Fuel Beds
The regression heads predict consumed fuel, not future fuel directly.

Reconstruction: 

```text
predicted future surface fuel = current surface fuel - predicted surface consumed fuel
predicted future canopy fuel = current canopy fuel - predicted canopy consumed fuel
```

These reconstructed maps are clamped to `>= 0`.

## Multi-Step Autoregressive Rollout
The default paper task is now a direct `prediction_horizon: 10` forecast. Use the standard test/evaluation scripts for that H-step target.

Autoregressive rollout remains available only for separate `prediction_horizon: 1` experiments: start from a true input window, predict one step ahead, write the predicted surface and canopy fuel back into the next raw frame, rebuild engineered features from that updated raw window, and repeat.

The rollout path currently supports only `window_mode: static`, which keeps the model input length equal to `input_sequence_length`. After every step, the oldest frame is dropped and the newly constructed next frame is appended.

Exogenous handling modes:
- `teacher_forced`: use the true future raw frame for non-fuel variables, but replace the surface and canopy fuel channels with predicted fuel
- `constant`: copy the previous rollout raw frame for non-fuel variables and replace the surface and canopy fuel channels with predicted fuel

The rollout script now randomly selects split samples by seed and generates one GIF per sample. Each GIF compares ground-truth future progression against the model's autoregressive rollout with consistent color scaling across steps. Metrics are still saved, but the primary output is the visual rollout sequence under `outputs/rollouts/<split>/<dataset_name>/`.

## Commands
All commands below assume the Conda environment above is already active.

### Core Python Scripts
- Inspect dataset split and file counts:
  `python scripts/inspect_dataset.py --config configs/default.yaml`
- Precompute train patch shards on scratch:
  `python scripts/precompute_patch_cache.py --config configs/default.yaml --split train`
- Compute normalization from the precomputed train patch cache:
  `python scripts/compute_normalization.py --config configs/default.yaml --from_cache`
- Precompute validation and test patch shards:
  `python scripts/precompute_patch_cache.py --config configs/default.yaml --split val`
  `python scripts/precompute_patch_cache.py --config configs/default.yaml --split test`
- Inspect the patch cache and save random previews:
  `python scripts/inspect_patch_cache.py --config configs/default.yaml --split all`
- Run the main project sanity check:
  `python scripts/sanity_check_project.py --config configs/default.yaml`
- Run lightweight smoke checks:
  `python scripts/smoke_checks.py --config configs/default.yaml`
- Train the model:
  `python scripts/train_forecasting_model.py --config configs/default.yaml`
- Evaluate the saved checkpoint on the configured test split:
  `python scripts/test_model.py --config configs/default.yaml --checkpoint-kind best`
- Probe model support for native and alternate spatial sizes without loading dataset files:
  `python scripts/test_spatial_size_compatibility.py --config configs/default.yaml`
- Diagnose why multitask mask output channel 2 generalizes poorly on external test:
  `python scripts/diagnose_mask_generalization.py --config configs/default.yaml --max_samples 200`
- Compare raw active-flux and burned-fuel mask definitions across splits:
  `python scripts/compare_mask_definitions.py --config configs/default.yaml --max_samples 200`
- Evaluate masks derived from predicted consumed fuel:
  `python scripts/evaluate_consumed_fuel_derived_mask.py --config configs/default.yaml --max_samples 200`
- Visualize validation predictions:
  `python scripts/visualize_predictions.py --config configs/default.yaml --split val --num_samples 10`
- Visualize test predictions:
  `python scripts/visualize_predictions.py --config configs/default.yaml --split test --num_samples 10`
- Visualize model vs persistence comparisons:

  `python scripts/visualize_model_vs_persistence.py --config configs/default.yaml --num_samples 20 --output_dir outputs/model_vs_persistence`
- Reconstruct future surface/canopy fuel beds from multitask predictions:

  `python scripts/reconstruct_fuel_bed_from_predictions.py --config configs/default.yaml --num_samples 10`
- Inspect engineered features for one sample:

  `python scripts/inspect_engineered_features.py --config configs/default.yaml --sample_index 0`
- Inspect atmospheric engineered features for one sample:

  `python scripts/inspect_atmospheric_features.py --config configs/default.yaml --sample_index 0`
- Legacy cache for engineered per-timestep tensors only:

  `python scripts/cache_engineered_dataset.py --config configs/default.yaml --output_dir ../keepz_05_engineered`

### Inspection And Analysis Scripts
- Inspect selected raw channel maps at selected timesteps:
  `python scripts/inspect_target_channels.py --config configs/default.yaml --channels 80 81 84 85 --timesteps 0 50 100`
- Launch the interactive raw input viewer:

  `python scripts/visualize_input_dataset.py --data-dir`
- Evaluate the persistence baseline on the configured test dataset:

  `python scripts/evaluate_persistence_baseline.py --config configs/default.yaml --num-visualizations 5`
- Compare persistence across candidate target channels:

  `python scripts/evaluate_persistence_all_candidate_targets.py --config configs/default.yaml --channels 50 51 52 53 54 55`
- Run multitask autoregressive rollout GIF generation with teacher-forced future exogenous variables:

  `python scripts/rollout_predictions.py --config configs/default.yaml --split test --num_samples 5 --rollout_steps 20`
- Run multitask autoregressive rollout GIF generation with constant exogenous variables:

  `python scripts/rollout_predictions.py --config configs/default.yaml --split test --num_samples 5 --rollout_steps 20 --exogenous_mode constant`

### Shell Wrappers
- Full pipeline wrapper:
  `bash scripts/run_full_pipeline.sh configs/default.yaml`
- Quick smoke-test wrapper:
  `bash scripts/run_quick_smoke_test.sh configs/default.yaml`
- Training-only wrapper:
  `bash scripts/run_training_only.sh configs/default.yaml`
- Visualization-only wrapper:
  `bash scripts/run_visualization_only.sh configs/default.yaml 10`

Use `--help` on any Python script for the full CLI:

```bash
python scripts/visualize_predictions.py --help
```

## FLARE v1 End-to-End
FLARE v1 is the first trainable end-to-end baseline for the fresh architecture. It uses four encoders, fire-query modality fusion, a small temporal CNN backbone, a shallow same-resolution decoder, and four heads.

Outputs are surface consumed fuel, canopy consumed fuel, fire mask logits, and log1p energy release. The v1 loss uses surface Huber weight 1, canopy Huber weight 1, mask BCE+Dice weight 5, energy-log Huber weight 1, and one auxiliary fire-support mask loss after local fused features with weight 0.2. There are no per-encoder auxiliary losses and no heavy backbone, neural operator, Mamba, or large transformer yet.

Smoke-test v1 with:

```bash
python scripts/smoke_test_cawfe_latte.py --config configs/default.yaml
```

Train with the example experiment config:

```bash
python scripts/train_forecasting_model.py \
  --config configs/experiments/cawfe_latte_v1.yaml
```

Evaluate with:

```bash
python scripts/evaluate_trained_models.py \
  --config configs/experiments/cawfe_latte_v1.yaml \
  --mode quantitative \
  --split test \
  --model_architecture cawfe_latte
```

## Archived CAWFE-Latte variants

Experimental CAWFE-Latte v1.1/v1.2/v1.3 variants were archived after poor validation generalization compared with the original CAWFE-Latte v1. They are not active architectures. Future CAWFE-Latte variants should be reintroduced progressively with one isolated change at a time.

Archived implementation/config/script/test snapshots live under `archive/failed_cawfe_latte_variants/`.

## Rebuilt Dataset Pipeline

The staged pipeline preserves the manual train/val/test fire splits from configuration and saves engineered full-frame tensors in channel-first `(C,H,W)` format under scratch. It does not construct targets or fixed X/y patch samples.

```bash
python scripts/build_engineered_frame_dataset.py --config configs/default.yaml  --skip_existing
python scripts/visualize_engineered_frames.py --config configs/default.yaml --split train
python scripts/build_patch_index.py --config configs/default.yaml
python scripts/visualize_patch_index.py --config configs/default.yaml --split train
```

## Rebuilt target/sample pipeline

After full-frame engineering and patch-index creation, build full-frame targets and metadata-only temporal samples:

```bash
python scripts/estimate_fire_mask_thresholds.py --config configs/default.yaml  --update_config --derived_config_path configs/derived/default_with_fire_mask_thresholds.yaml
python scripts/build_target_dataset.py --config configs/derived/default_with_fire_mask_thresholds.yaml
python scripts/visualize_targets.py --config configs/default.yaml --split train
python scripts/build_temporal_sample_index.py --config configs/default.yaml --pattern all
python scripts/visualize_processed_samples.py --config configs/default.yaml --pattern single1_h10 --split train
python scripts/compute_processed_dataset_normalization.py --config configs/default.yaml --pattern single1_h10
python scripts/inspect_processed_dataset.py --config configs/default.yaml
```

```bash
python scripts/train_forecasting_model.py --config configs/experiments/convlstm_consecutive5_h10.yaml
python scripts/train_forecasting_model.py --config configs/experiments/convlstm_single1_h10.yaml
python scripts/train_forecasting_model.py --config configs/experiments/convlstm_sparse5_h10.yaml
```

## Automated Data Preparation Pipeline

Run the rebuilt data stages in order with one command:

```bash
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml --make-quicklooks
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml --percentile 5.0
bash scripts/run_data_preparation_pipeline.sh configs/default.yaml --pattern sparse5_h10
```

The runner builds engineered frames, estimates train-only fire-mask thresholds, constructs targets using the frozen derived config, builds patch and temporal indices, computes train-only normalization, inspects the dataset, and optionally saves visualizations. It never trains models by default. Logs are written under `artifacts/logs/data_preparation/`; after success, the script prints the exact next training commands.

## Terrain Processing and FLARE Conditioning

When `terrain.enabled: true`, the engineered-frame builder finds each fire's `.terrain` file, validates its dimensions against the processed frames, and saves it under `fires/<FIRE_NAME>/terrain/`. It stores the original file, elevation, and four static feature maps: relative elevation, slope magnitude, `slope_x`, and `slope_y`.

The processed dataloader crops the same terrain patch as the dynamic inputs and targets. Terrain is returned as `B x 4 x H x W` without a time dimension. FLARE encodes these maps with a CNN and applies zero-initialized FiLM conditioning after dynamic fire-query cross-attention; terrain is not treated as a dynamic attention token. ConvLSTM and other architectures continue to receive the existing tuple `(X, y)` unless terrain is explicitly enabled.

Build and inspect terrain with:

```bash
python scripts/build_engineered_frame_dataset.py --config configs/default.yaml
python scripts/visualize_terrain.py --config configs/default.yaml --split train
python scripts/visualize_processed_samples.py \
  --config configs/experiments/cawfe_latte_v1.yaml \
  --pattern consecutive5_h10 --split train
```

## FLARE Ablations

FLARE has one active architecture, `model.architecture: cawfe_latte`. The first 10-epoch screening batch changes only the post-fusion backbone (A/B) or temporal pooling (C); early stopping and automatic test evaluation are disabled.

```bash
python scripts/check_cawfe_latte_ablation_configs.py
bash scripts/submit_cawfe_latte_ablation_batch.sh \
  baseline A_resblocks B_multiscale_context C_temporal_attention
python scripts/summarize_cawfe_latte_ablations.py
```

Submit the second screening batch with the same generic Slurm workflow:

```bash
bash scripts/submit_cawfe_latte_ablation_batch.sh \
  G_separate_regression_decoder \
  I_patch_fire_classifier \
  K_no_terrain_film \
  L_simple_concat_fusion
```

Each standard ablation job now performs both validation phases automatically:
a shared deterministic fire/no-fire-stratified 50-batch subset every epoch, then
one complete validation pass with `checkpoints/best_model.pt` after training.
The batch launcher prepares/reuses the shared screening index before allocating
GPUs. Each run exits only after writing `screening_validation_metrics.json`,
`full_validation_metrics.json`, and `full_validation_per_fire.json`.

After every submitted training job finishes, the only required reporting step is:

```bash
python scripts/summarize_cawfe_latte_ablations.py
```

`scripts/recompute_ablation_no_fire_metrics.py` is retained only as a recovery
tool for historical runs. It is not invoked by the standard submission, training,
or summarization workflow.

## Per-epoch FLARE fusion vectors

Optionally save one representative post-fusion vector at the start of each epoch:

```yaml
training:
  save_fusion_vectors:
    enabled: true
    output_name: fusion_vectors_by_epoch.npy
    vector_reduce: mean_bt_hw
```

The run writes `features/fusion_vectors_by_epoch.npy` with shape `(num_epochs_completed, fusion_dim)`, plus metadata JSON by default.
