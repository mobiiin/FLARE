from __future__ import annotations

import numpy as np

from scripts import plot_background_weak_fire as plot


def test_selection_uses_canonical_bins_and_observed_nearby_fire(monkeypatch):
    def record(sample_id, fire, current, target, x0, y0):
        return {'sample_id': sample_id, 'fire_name': fire, 'split': 'test',
                'current_index': current, 'target_index': target,
                'patch': {'x0': x0, 'y0': y0, 'width': 64, 'height': 64}}

    records = [record('near_no_fire', 'A', 10, 20, 0, 0),
               record('far_no_fire', 'A', 10, 20, 128, 128),
               record('past_active', 'A', 0, 10, 64, 0),
               record('weak', 'B', 10, 20, 0, 0)]
    activity = {
        'near_no_fire': {'activity_bin': 'no_fire', 'active_fraction': '0'},
        'far_no_fire': {'activity_bin': 'no_fire', 'active_fraction': '0'},
        'past_active': {'activity_bin': 'tiny_fire', 'active_fraction': str(1/4096)},
        'weak': {'activity_bin': 'tiny_fire', 'active_fraction': str(1/4096)},
    }

    def raw_case(root, item, **kwargs):
        target = np.zeros((4, 64, 64), dtype=np.float32)
        if item['sample_id'] in ('past_active', 'weak'):
            target[2, 1, 1] = 1
        return None, target, None

    monkeypatch.setattr(plot, 'raw_case', raw_case)
    monkeypatch.setattr(plot, 'case_metadata', lambda root, item: {
        'sample_id': item['sample_id'], 'fire_name': item['fire_name'],
        'split': item['split'], 'patch': item['patch'],
    })

    class Observed:
        def predict_one(self, item):
            return np.zeros((4, 64, 64), dtype=np.float32)

    cases = plot.select_pair(None, records, activity, {}, 7, Observed())
    assert [case['sample_id'] for case in cases] == ['near_no_fire', 'weak']
    assert [case['target_activity_bin'] for case in cases] == ['no_fire', 'tiny_fire']
    assert cases[0]['nearby_observed_fire'] is True


def test_color_limit_is_shared_across_models_and_tight_for_no_fire():
    target = np.zeros((4, 2, 2), dtype=np.float32)
    target[0] = 100
    predictions = {}
    for key, value in zip(plot.MODEL_KEYS, (1, 2, 3)):
        raw = np.zeros_like(target)
        raw[0] = value
        predictions[key] = raw
    data = {'target': target, 'predictions': predictions}
    no_fire = plot._figure_limits({'case_type': 'no_fire'}, data, 0)
    weak = plot._figure_limits({'case_type': 'weak_fire'}, data, 0)
    assert 2 < no_fire < 4
    assert weak > 90
