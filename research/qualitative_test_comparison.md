# FLARE qualitative comparisons

Run one command from the repository root in the `fire_forecasting` environment. The second argument is the **number of PNG figures per fire**:

```bash
python scripts/qualitative_test_comparison.py CHIMNEYTOPS2 10
python scripts/qualitative_test_comparison.py test 10
python scripts/qualitative_test_comparison.py val 10
python scripts/qualitative_test_comparison.py train 10
python scripts/qualitative_test_comparison.py all 10
```

A fire name selects that fire in its assigned dataset split. `test`, `val`, and `train` select every fire in that split. `all` selects every fire in all three splits. The current index contains 7 test fires, 6 validation fires, and 22 training fires, so `all 10` generates 350 PNGs. Fire names and split keywords are case-insensitive.

Figures are saved under `artifacts/qualitative_test_comparison/<split>/<fire>/`. Each PNG compares ground truth, ConvLSTM U-Net, FLARE baseline, and FLARE final. The fire-mask row retains model probabilities. For surface fuel consumed, canopy fuel consumed, and log energy, each model panel shows the **absolute error** `|prediction − ground truth|` instead of the raw prediction. The three models share one error color scale within each row; ground truth keeps its own scale. The left context column shows observed fire, timestamp information, and the **full, uncropped** observed surface-fuel map from raw channel 84 at `t`; an orange box marks the evaluated patch. The four row labels are Active fire, Surface fuel, Canopy fuel, and HRR. Surface/Canopy fuel rows still represent interval fuel consumption, and HRR is shown in log(1+MW) units. Wind arrows in this per-fire figure show direction at a fixed visual length, while their raw components still come from the configured low levels. The figure title identifies its split. The command creates no selection sheets, PDFs, SVGs, metrics files, or manifests.

For each figure, the script generates a fresh random selection seed, chooses a distinct observed timestamp within that fire and split, and picks one random patch at that timestamp with at least one active future ground-truth fire pixel. Selection uses ground truth only and does not inspect model predictions or errors. The generated seed, timestamp, and sample ID are printed; the seed and timestamp are also included in the PNG filename, which ends in `_abs_error_full_fuel.png`. The three learned models use their frozen Table 2 **seed 42** checkpoints for every selected patch.

Training and validation figures show those splits. Use test figures when presenting held-out test performance.
