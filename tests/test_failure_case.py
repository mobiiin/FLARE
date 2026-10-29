from __future__ import annotations

from scripts.plot_failure_case import choose_failure, rank_failures


def _row(sample_id, activity_bin, dice, hrr, active_canopy, surface=0.0, canopy=0.0):
    return {'sample_id': sample_id, 'fire_name': 'FIRE', 'activity_bin': activity_bin,
            'dice': dice, 'energy_log_mae': hrr, 'active_canopy_mae': active_canopy,
            'surface_mae': surface, 'canopy_mae': canopy}


def test_default_rank_excludes_no_fire_and_tiny_and_uses_metric_direction():
    rows = [_row('no', 'no_fire', None, 100, None),
            _row('tiny', 'tiny_fire', 0, 90, 90),
            _row('small', 'small_fire', .2, 1, 1),
            _row('medium', 'medium_fire', .5, 3, 3),
            _row('large', 'large_fire', .8, 2, 2)]
    assert [row['sample_id'] for row in rank_failures(rows, 'dice')] == ['small', 'medium', 'large']
    assert [row['sample_id'] for row in rank_failures(rows, 'hrr')] == ['medium', 'large', 'small']
    assert rank_failures(rows, 'hrr', allow_no_fire=True)[0]['sample_id'] == 'no'


def test_composite_averages_percentile_ranks_and_seeded_pool_is_unique():
    rows = [_row('a', 'medium_fire', 0, 1, 1),
            _row('b', 'medium_fire', .5, 1000, 1000),
            _row('c', 'medium_fire', 1, 1001, 1001)]
    ranked = rank_failures(rows, 'composite')
    assert [row['sample_id'] for row in ranked] == ['c', 'b', 'a']
    assert abs(ranked[0]['criterion_value'] - 2/3) < 1e-12
    assert choose_failure(ranked, 2, 99, rank=1)['sample_id'] == 'c'

    pool = [{'sample_id': str(index), 'test_rank': index + 1} for index in range(10)]
    picks = [choose_failure(pool, 10, seed)['sample_id'] for seed in range(10)]
    assert len(set(picks)) == 10


def test_unified_png_plan_balances_rim_thomas_and_avoids_duplicate_cases():
    from scripts.failure_cases import FIRES, plan_failures

    rows = []
    for fire in ('rim', 'thomas'):
        for index in range(6):
            row = _row(f'{fire}_{index}', 'medium_fire', .1 + .1*index,
                       6 - index, 6 - index)
            row['fire_name'] = FIRES[fire]
            rows.append(row)
    plan = plan_failures(rows, 10, top_k=6, selection_seed=42)
    assert len(plan) == len({row['sample_id'] for row in plan}) == 10
    assert [row['fire_short'] for row in plan[:8]] == ['rim', 'thomas'] * 4
    assert [row['criterion'] for row in plan[:8]] == [
        'composite', 'composite', 'dice', 'dice', 'hrr', 'hrr',
        'active_canopy', 'active_canopy',
    ]
    assert sum(row['fire_short'] == 'rim' for row in plan) == 5
    assert sum(row['fire_short'] == 'thomas' for row in plan) == 5
    assert plan == plan_failures(rows, 10, top_k=6, selection_seed=42)
