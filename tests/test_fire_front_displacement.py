from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation import fire_front_displacement as front


def test_current_mask_uses_only_preceding_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {'sample_id': 'example', 'fire_name': 'FIRE', 'current_index': 20,
              'horizon': 10, 'patch': {'y0': 1, 'x0': 1, 'height': 2, 'width': 2}}
    loaded = []

    def frame(path: str) -> np.ndarray:
        loaded.append(Path(path).name)
        return np.full((86, 4, 4), len(loaded), dtype=np.float32)

    def build(previous, current, area, config, thresholds):
        assert np.all(previous == 1)
        assert np.all(current == 2)
        return {'fire_mask': np.array([[0, 0, 0, 0], [0, 1, 1, 0],
                                      [0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)}

    monkeypatch.setattr(front, '_raw_frame', frame)
    monkeypatch.setattr(front, '_area', lambda root, fire: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(front, 'build_processed_target', build)
    mask = front.reconstruct_current_mask(Path('/data'), record, {}, {})

    assert loaded == ['frame_000010.npz', 'frame_000020.npz']
    np.testing.assert_array_equal(mask, [[True, True], [True, False]])


def test_boundary_metrics_are_zero_for_identical_masks() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:11, 5:12] = True
    assert front.boundary_metrics(mask, mask) == {
        'symmetric_mean_boundary_distance_pixels': 0.0,
        'hausdorff_distance_pixels': 0.0,
        'centroid_error_pixels': 0.0,
    }
