from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.evaluation.qualitative_test_figure import (
    absolute_error_limit, absolute_error_maps, draw_error_comparison_png, observed_context,
)


def test_absolute_error_uses_unclipped_predictions_and_one_shared_scale() -> None:
    target = np.zeros((4, 2, 2), dtype=np.float32)
    target[0, 0, 0] = 1.0
    predictions = {name: target.copy() for name in ('convlstm', 'baseline', 'final')}
    predictions['convlstm'][0, 0, 0] = -2.0
    predictions['baseline'][0, 0, 0] = 1.5
    predictions['final'][0, 0, 0] = 1.25

    errors = absolute_error_maps(target, predictions, tuple(predictions), 0)

    assert errors['convlstm'][0, 0] == 3.0  # Negative predictions must count.
    assert errors['baseline'][0, 0] == 0.5
    assert errors['final'][0, 0] == 0.25
    assert absolute_error_limit(errors, 100) == 3.0


def test_observed_surface_fuel_keeps_full_raw_frame_and_patch(tmp_path: Path) -> None:
    frame_dir = tmp_path / 'fires' / 'FIRE' / 'frames'
    frame_dir.mkdir(parents=True)
    frame = np.zeros((86, 6, 6), dtype=np.float32)
    frame[84] = np.arange(36, dtype=np.float32).reshape(6, 6)
    np.savez_compressed(frame_dir / 'frame_000020.npz', x_raw=frame)
    record = {'fire_name': 'FIRE', 'current_index': 20,
              'patch': {'y0': 1, 'x0': 2, 'height': 3, 'width': 2}}
    config = {'multitask': {'surface_fuel_channel': 84},
              'atmospheric_features': {'low_level_indices': [0, 1, 2], 'variables_per_level': 10}}

    class Predictor:
        def predict_one(self, record):
            return np.zeros((4, 3, 2), dtype=np.float32)

    context = observed_context(tmp_path, record, config, Predictor())
    np.testing.assert_array_equal(context['surface_fuel_full'], frame[84])
    np.testing.assert_array_equal(context['surface_fuel'], frame[84, 1:4, 2:4])
    assert context['surface_fuel_patch_bounds'] == {'y0': 1, 'x0': 2, 'height': 3, 'width': 2}
    assert context['surface_fuel_channel'] == 84


def test_error_figure_writes_one_png(tmp_path: Path) -> None:
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/plots/qualitative_test_comparison.yaml').read_text())
    config['figure']['dpi'] = 80
    config['wind']['enabled'] = False
    config['terrain']['contours'] = False
    config['_wind_levels'] = [0, 1, 2]
    target = np.zeros((4, 8, 8), dtype=np.float32)
    target[0, 2:5, 2:5] = .3
    target[1, 3:6, 3:6] = .1
    target[2, 2:6, 2:6] = 1
    target[3, 2:6, 2:6] = .5
    predictions = {name: target.copy() for name in ('convlstm', 'baseline', 'final')}
    predictions['convlstm'][0, 2:5, 2:5] = -.1
    case = {'sample_id': 'sample', 'split': 'test', 'fire_name': 'FIRE',
            'latest_input_timestamp': 20, 'target_timestamp': 30, 'target_active_fraction': .25}
    full_fuel = np.arange(20*16, dtype=np.float32).reshape(20, 16)
    context = {'surface_fuel': full_fuel[4:12, 5:13],
               'surface_fuel_full': full_fuel,
               'surface_fuel_patch_bounds': {'y0': 4, 'x0': 5, 'height': 8, 'width': 8},
               'surface_fuel_channel': 84,
               'latest_observed_fire_state': np.zeros((8, 8), dtype=np.float32),
               'wind_u': np.zeros((8, 8), dtype=np.float32),
               'wind_v': np.zeros((8, 8), dtype=np.float32), 'terrain_elevation': None}
    output = tmp_path / 'comparison.png'

    draw_error_comparison_png(case, {'target': target, 'context': context, 'predictions': predictions}, output, config)
    assert output.is_file() and output.stat().st_size > 0
    assert list(tmp_path.glob('*.pdf')) == []
