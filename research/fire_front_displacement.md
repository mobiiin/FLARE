# Held-out-test burn-footprint boundary figure

The paper-facing model names in this figure are **FLARE baseline** and **FLARE final** (the GA_Q2 finalist). The plotted contours are boundaries of **interval burn footprints**. They are not instantaneous simulator fire perimeters.

## Reproduce

Run on a machine with the existing processed dataset and frozen Table 2 run directories. The figure command uses CPU inference only and does not submit jobs or train models.

```bash
python scripts/plot_fire_front_displacement.py --selection-seed 42 --selection-only
python scripts/plot_fire_front_displacement.py \
  --freeze-selection artifacts/fire_front_displacement/selection_seed_42/selected_cases.json \
  --model-seed 42
```

The first command saves `selected_cases.json` and `selection_preview.png` using test inputs, terrain, wind, and ground truth only. The selection is written before any checkpoint is opened. Inspect and freeze that file before using the second command. For several GT-only previews:

```bash
python scripts/generate_front_displacement_candidates.py \
  --selection-seeds 42 123 2026 7 11 --selection-only
```

The second command resolves baseline and final checkpoints for the *same* frozen Table 2 seed (42, 123, or 2026), runs both on the same three test crops, and saves PDF, SVG, 600-dpi PNG, `panel_metrics.csv`, `arrays.npz`, `figure_metadata.json`, and `checkpoint_manifest.json` under `artifacts/fire_front_displacement/selection_seed_42/model_seed_42/`. To redraw after inference, use `--render-only` with the same selection and seed; this reads the saved arrays without loading checkpoints. `--terrain on/off`, `--wind on/off`, `--show-centroid-arrow`, `--annotate-metrics`, and `--config` control appearance. Figure and selection settings live in `configs/plots/fire_front_displacement.yaml`.

## Data definitions

The current boundary comes from the canonical target builder applied to raw frames `[t-10,t]`, using interval surface and canopy consumption plus heat-release power at `t`. The future ground-truth boundary is canonical target channel 2 for `[t,t+10]`, thresholded at `>0.5`. The predicted boundary is channel-2 mask logits after sigmoid, also thresholded at `>0.5`. The script verifies the rebuilt future mask and fuel channels exactly against the stored target; the energy channel allows at most `1e-6` absolute float32 rounding. All three contours use the same binary-mask boundary definition.

Case selection uses only GT/input-derived descriptors. It requires three different held-out test fires and chooses directional, curved/multi-front, and challenging-physical cases. The current mask is reconstructed from frames at or before `t`; wind is drawn from the latest observed raw frame (direction-normalized arrows by default; raw U/V arrays are saved) and terrain from the fire's relative-elevation feature. Boundary distances are reported in grid cells, with no assumed meter conversion. Dice and IoU use the fixed evaluation threshold and remain the official overlap metrics; boundary metrics are diagnostics for this figure.
