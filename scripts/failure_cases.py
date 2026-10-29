#!/usr/bin/env python3
"""Make a PNG gallery of objective FLARE final test failures from RIM and Thomas.

Simple use: python scripts/failure_cases.py 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_qualitative_forecasts import case_metadata
from scripts.plot_qualitative_test_comparison import _data_for_cases, _load_one_model, resolve_figure_run
from scripts.plot_failure_case import _metric_path, draw_failure_figure, load_verified_metrics, rank_failures
from src.config import load_config
from src.evaluation.qualitative_test_figure import read_test_records

OUTPUT_ROOT = ROOT / 'artifacts/failure_cases/png_gallery'
FIRES = {'rim': 'RIM__0822__keepz_08', 'thomas': 'THOMAS__1210__keepz_08'}
DEFAULT_CRITERIA = ('composite', 'dice', 'hrr', 'active_canopy')
OPTIONAL_CRITERIA = ('surface', 'canopy')


def plan_failures(rows, figure_count, *, fires=('rim', 'thomas'), criteria=DEFAULT_CRITERIA,
                  top_k=50, selection_seed=42):
    """Choose distinct cases: worst per fire/criterion, then seeded top-k cases."""
    if figure_count < 1 or top_k < 1 or not fires or not criteria:
        raise ValueError('figure_count and top_k must be positive; select fires and criteria.')
    ranked_by_pair = {}
    pairs = [(fire, criterion) for criterion in criteria for fire in fires]
    for criterion in criteria:
        global_ranking = rank_failures(rows, criterion)
        for fire in fires:
            fire_rows = [row for row in global_ranking if row['fire_name'] == FIRES[fire]]
            if not fire_rows:
                raise RuntimeError(f'No eligible test cases for {fire}/{criterion}')
            ranked_by_pair[(fire, criterion)] = [
                {**row, 'global_test_rank': row['test_rank'], 'test_rank': position,
                 'rank_label': 'fire rank', 'criterion': criterion, 'fire_short': fire,
                 'selection_mode': 'highest_available'}
                for position, row in enumerate(fire_rows, start=1)]
    chosen = []
    used = set()
    # First include each requested failure mode and fire, using the worst
    # available case if another criterion already selected the same sample.
    for pair in pairs:
        row = next((item for item in ranked_by_pair[pair] if item['sample_id'] not in used), None)
        if row is None:
            continue
        chosen.append(row)
        used.add(row['sample_id'])
        if len(chosen) == figure_count:
            return chosen
    # Further candidates are sampled from each fire/criterion's objective top-k
    # using a deterministic random order; model predictions are not inspected.
    shuffled = {}
    for fire, criterion in pairs:
        pool = ranked_by_pair[(fire, criterion)][:top_k].copy()
        random.Random(f'{selection_seed}:{fire}:{criterion}').shuffle(pool)
        shuffled[(fire, criterion)] = pool
    while len(chosen) < figure_count:
        added = False
        for pair in pairs:
            row = next((item for item in shuffled[pair] if item['sample_id'] not in used), None)
            if row is None:
                continue
            chosen.append({**row, 'selection_mode': 'seeded_top_k'})
            used.add(row['sample_id'])
            added = True
            if len(chosen) == figure_count:
                break
        if not added:
            raise ValueError(f'Only {len(chosen)} distinct candidates are available in the requested top-{top_k} pools.')
    return chosen


def gallery_dir(*, figure_count, fires, criteria, selection_seed, model_seed, top_k):
    fire_label = '_'.join(fires)
    criterion_label = '_'.join(criteria)
    return OUTPUT_ROOT / (f'{fire_label}_{criterion_label}_n{figure_count}'
                          f'_top{top_k}_selection{selection_seed}_model{model_seed}')


def generate_pngs(figure_count, *, fire='both', criterion='all', top_k=50,
                  selection_seed=42, model_seed=42):
    if model_seed not in (42, 123, 2026):
        raise ValueError('model_seed must be 42, 123, or 2026')
    fires = tuple(FIRES) if fire == 'both' else (fire,)
    if any(name not in FIRES for name in fires):
        raise ValueError('fire must be both, rim, or thomas')
    criteria = DEFAULT_CRITERIA if criterion == 'all' else (criterion,)
    if any(name not in (*DEFAULT_CRITERIA, *OPTIONAL_CRITERIA) for name in criteria):
        raise ValueError(f'Unknown criterion: {criterion}')
    torch.set_num_threads(2)
    config = load_config(ROOT / 'artifacts/table2_baselines/persistence/resolved_config.yaml')
    root, _, records, _ = read_test_records(config)
    by_id = {record['sample_id']: record for record in records}
    run = resolve_figure_run('final', model_seed)
    if run['signature'][-1] != len(records):
        raise RuntimeError('Frozen checkpoint and held-out test index disagree.')
    metric_path = _metric_path(run, model_seed)
    try:
        metrics = load_verified_metrics(metric_path, records, model_seed)
    except (RuntimeError, ValueError):
        metric_path = _metric_path(run, model_seed, use_original=False, recompute=True)
        metrics = load_verified_metrics(metric_path, records, model_seed)
    selected = plan_failures(metrics, figure_count, fires=fires, criteria=criteria,
                             top_k=top_k, selection_seed=selection_seed)
    cases = []
    for row in selected:
        record = by_id[row['sample_id']]
        case = case_metadata(root, record)
        case.update({'case_type': 'selected', 'target_activity_bin': row['activity_bin'],
                     'target_active_fraction': row['active_fraction']})
        row['latest_input_timestamp'] = case['latest_input_timestamp']
        row['target_timestamp'] = case['target_timestamp']
        cases.append(case)
    data = _data_for_cases(root, cases, by_id, config)
    _load_one_model(run, cases, by_id, data, {})
    output = gallery_dir(figure_count=figure_count, fires=fires, criteria=criteria,
                         selection_seed=selection_seed, model_seed=model_seed, top_k=top_k)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, (row, panel) in enumerate(zip(selected, data), start=1):
        filename = (f'{number:03d}_{row["fire_short"]}_{row["criterion"]}'
                    f'_rank{row["test_rank"]:03d}_t{row["latest_input_timestamp"]:06d}')
        draw_failure_figure(row, panel, output, show_error_map=True, layout='compact',
                            formats=('png',), filename=filename)
        path = output / f'{filename}.png'
        paths.append(path)
        print(f'FIGURE {number}/{figure_count}: {row["fire_short"].upper()} | '
              f'{row["criterion"]} | fire rank {row["test_rank"]} | '
              f'global test rank {row["global_test_rank"]} | {path}', flush=True)
    print(f'PNG FIGURES: {len(paths)}')
    print(f'OUTPUT DIRECTORY: {output}')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure_count', type=int, help='Total number of PNG figures to generate')
    parser.add_argument('--fire', choices=('both', 'rim', 'thomas'), default='both',
                        help='Fire selection; default: both RIM and Thomas')
    parser.add_argument('--criterion', choices=('all', *DEFAULT_CRITERIA, *OPTIONAL_CRITERIA),
                        default='all', help='Failure metric; default: rotate four paper criteria')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--model-seed', type=int, choices=(42, 123, 2026), default=42)
    args = parser.parse_args(argv)
    if args.figure_count < 1 or args.top_k < 1:
        parser.error('figure_count and --top-k must be positive')
    return generate_pngs(args.figure_count, fire=args.fire, criterion=args.criterion,
                         top_k=args.top_k, selection_seed=args.selection_seed,
                         model_seed=args.model_seed)


if __name__ == '__main__':
    main()
