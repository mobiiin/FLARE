#!/usr/bin/env python3
"""Objectively rank held-out FLARE final failures and draw CPU-only paper figures."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_qualitative_forecasts import case_metadata
from scripts.plot_qualitative_test_comparison import (
    _data_for_cases, _load_one_model, json_write, resolve_figure_run,
)
from src.config import load_config
from src.evaluation.fire_activity import active_fraction_bin_name
from src.evaluation.full_validation import _sample_metric_rows
from src.evaluation.qualitative_test_figure import color_limits, gt_activity_index, read_test_records, sha256_file

OUTPUT_ROOT = ROOT / 'artifacts/failure_cases'
DATASET_CONFIG = ROOT / 'artifacts/table2_baselines/persistence/resolved_config.yaml'
CRITERIA = ('dice', 'hrr', 'active_canopy', 'surface', 'canopy', 'composite')
METRIC_KEYS = {'dice': 'dice', 'hrr': 'energy_log_mae', 'active_canopy': 'active_canopy_mae',
               'surface': 'surface_mae', 'canopy': 'canopy_mae'}
DISPLAY_NAMES = {'dice': 'Dice', 'hrr': 'HRR Log MAE', 'active_canopy': 'Active Canopy MAE',
                 'surface': 'Surface MAE', 'canopy': 'Canopy MAE', 'composite': 'Composite score'}
PANEL_METRICS = {'Dice': 'dice', 'IoU': 'iou', 'Surface MAE': 'surface_mae',
                 'Canopy MAE': 'canopy_mae', 'HRR Log MAE': 'energy_log_mae',
                 'Active Canopy MAE': 'active_canopy_mae'}
ROWS = (('Active fire', 2, 'magma'), ('Surface fuel', 0, 'viridis'),
        ('Canopy fuel', 1, 'cividis'), ('HRR', 3, 'inferno'))


def _metric_path(run, model_seed, *, use_original=True, recompute=False):
    evaluation = run['run_dir'] / 'evaluation'
    if use_original:
        for suffix in ('csv', 'parquet'):
            path = evaluation / f'test_sample_metrics.{suffix}'
            if path.is_file():
                return path
    cache = OUTPUT_ROOT / '_sample_metrics_cache' / f'model_seed_{model_seed}'
    if not recompute:
        for suffix in ('csv', 'parquet'):
            path = cache / 'evaluation' / f'test_sample_metrics.{suffix}'
            if path.is_file():
                return path
    print('Sample-level test metrics missing; running the frozen CPU test evaluator once.', flush=True)
    from scripts.evaluate_per_fire import _rerun
    from src.evaluation.flare_runs import resolve_run
    frozen = resolve_run('cawfe_latte_final', model_seed)
    if frozen['checkpoint'].resolve() != run['checkpoint'].resolve():
        raise RuntimeError('Fallback checkpoint differs from the frozen Table 2 checkpoint.')
    _rerun(frozen, cache)
    for suffix in ('csv', 'parquet'):
        path = cache / 'evaluation' / f'test_sample_metrics.{suffix}'
        if path.is_file():
            return path
    raise FileNotFoundError(f'CPU test evaluation did not save per-sample metrics under {cache}')


def _read_rows(path):
    if path.suffix == '.csv':
        with path.open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(f'Reading {path} requires pandas and a Parquet engine.') from exc
    return pd.read_parquet(path).to_dict(orient='records')


def _number(row, key):
    raw = row.get(key)
    if raw is None or raw == '':
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {key} for {row.get("sample_id")}: {raw}')
    return value


def load_verified_metrics(path, records, model_seed):
    """Accept only a complete, checkpoint-seed-matched, canonical test table."""
    source = _read_rows(path)
    by_id = {record['sample_id']: record for record in records}
    canonical = gt_activity_index(records)
    required = {'sample_id', 'fire_name', 'model_name', 'seed', 'activity_bin',
                'active_pixel_count', 'pixel_count', *PANEL_METRICS.values()}
    if not source or not required <= source[0].keys() or len(source) != len(records):
        raise RuntimeError(f'Incomplete FLARE final sample metrics: {path}')
    parsed = []
    seen = set()
    for row in source:
        sample_id = str(row['sample_id'])
        record = by_id.get(sample_id)
        if record is None or sample_id in seen:
            raise RuntimeError(f'Unexpected or duplicate test sample ID in {path}: {sample_id}')
        seen.add(sample_id)
        if (row['fire_name'] != record['fire_name'] or row['model_name'] != 'cawfe_latte_final'
                or int(row['seed']) != model_seed):
            raise RuntimeError(f'FLARE final model, seed, or fire mismatch for {sample_id}')
        pixels, active = int(row['pixel_count']), int(row['active_pixel_count'])
        fraction = active / pixels
        if (pixels <= 0 or active < 0 or active > pixels
                or row['activity_bin'] != active_fraction_bin_name(fraction)
                or row['activity_bin'] != canonical[sample_id]['activity_bin']
                or active != int(canonical[sample_id]['active_pixel_count'])):
            raise RuntimeError(f'Canonical activity mismatch for {sample_id}')
        item = {'sample_id': sample_id, 'fire_name': row['fire_name'],
                'activity_bin': row['activity_bin'], 'active_fraction': fraction,
                'active_pixel_count': active}
        for key in PANEL_METRICS.values():
            item[key] = _number(row, key)
        parsed.append(item)
    if seen != set(by_id):
        raise RuntimeError(f'Sample metric table does not cover the complete held-out test split: {path}')
    return parsed


def _badness_percentiles(rows, key, *, high_is_worse):
    values = sorted(row[key] for row in rows)
    denominator = max(len(values) - 1, 1)
    result = {}
    for row in rows:
        value = row[key]
        midpoint = (bisect_left(values, value) + bisect_right(values, value) - 1) / 2
        position = midpoint / denominator
        result[row['sample_id']] = position if high_is_worse else 1 - position
    return result


def rank_failures(rows, criterion, allow_no_fire=False):
    """Rank eligible samples from the complete saved test metric table."""
    if criterion not in CRITERIA:
        raise ValueError(f'Unknown criterion: {criterion}')
    allowed_bins = {'small_fire', 'medium_fire', 'large_fire'}
    if allow_no_fire:
        allowed_bins.update(('no_fire', 'tiny_fire'))
    required = ('dice', 'energy_log_mae', 'active_canopy_mae') if criterion == 'composite' else (METRIC_KEYS[criterion],)
    eligible = [row.copy() for row in rows if row['activity_bin'] in allowed_bins
                and all(row[key] is not None for key in required)]
    if not eligible:
        raise RuntimeError(f'No eligible held-out test samples for criterion={criterion}')
    if criterion == 'composite':
        dice = _badness_percentiles(eligible, 'dice', high_is_worse=False)
        hrr = _badness_percentiles(eligible, 'energy_log_mae', high_is_worse=True)
        canopy = _badness_percentiles(eligible, 'active_canopy_mae', high_is_worse=True)
        for row in eligible:
            sample_id = row['sample_id']
            row['criterion_value'] = (dice[sample_id] + hrr[sample_id] + canopy[sample_id]) / 3
        eligible.sort(key=lambda row: (-row['criterion_value'], row['sample_id']))
    else:
        key = METRIC_KEYS[criterion]
        high_is_worse = criterion != 'dice'
        for row in eligible:
            row['criterion_value'] = row[key]
        eligible.sort(key=lambda row: ((-row[key]) if high_is_worse else row[key], row['sample_id']))
    for rank, row in enumerate(eligible, start=1):
        row['test_rank'] = rank
    return eligible


def choose_failure(ranked, top_k, selection_seed, rank=None):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if rank is not None:
        if rank < 1 or rank > len(ranked):
            raise ValueError(f'--rank must be from 1 to {len(ranked)}')
        return ranked[rank - 1]
    pool = ranked[:min(top_k, len(ranked))]
    # Consecutive seeds sample a shuffled pool without replacement. Every
    # complete cycle receives a fresh, reproducible random permutation.
    cycle, position = divmod(int(selection_seed), len(pool))
    order = list(range(len(pool)))
    random.Random(cycle).shuffle(order)
    return pool[order[position]]


def _candidate_root(criterion, model_seed, top_k):
    output = OUTPUT_ROOT / criterion
    if model_seed != 42:
        output = output / f'model_seed_{model_seed}'
    if top_k != 50:
        output = output / f'top_k_{top_k}'
    return output


def _write_pool(path, ranked, top_k):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        fields = ('test_rank', 'sample_id', 'fire_name', 'activity_bin', 'criterion_value',
                  'dice', 'energy_log_mae', 'active_canopy_mae', 'surface_mae', 'canopy_mae')
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ranked[:min(top_k, len(ranked))])


def _save_arrays(path, data):
    target = np.asarray(data['target'], dtype=np.float32)
    prediction = np.asarray(data['predictions']['final'], dtype=np.float32)
    context = data['context']
    probability = (1 / (1 + np.exp(-np.clip(prediction[2], -60, 60)))).astype(np.float32)
    np.savez_compressed(path,
        gt_active_fire=target[2], gt_surface_fuel=target[0], gt_canopy_fuel=target[1],
        gt_hrr_log=target[3], flare_active_fire_probability=probability,
        flare_mask_logits=prediction[2], flare_surface_fuel=prediction[0],
        flare_canopy_fuel=prediction[1], flare_hrr_log=prediction[3],
        context_latest_observed_fire=np.asarray(context['latest_observed_fire_state'], dtype=np.float32),
        context_wind_u=np.asarray(context['wind_u'], dtype=np.float32),
        context_wind_v=np.asarray(context['wind_v'], dtype=np.float32))


def _save_panel_metrics(path, selection, computed):
    """Save metrics of the displayed CPU prediction and its Table 2 comparison."""
    row = {'sample_id': selection['sample_id'], 'fire_name': selection['fire_name'],
           'activity_bin': selection['activity_bin']}
    comparison = {}
    for label, key in PANEL_METRICS.items():
        expected, actual = selection[key], computed[key]
        row[label] = actual
        comparison[label] = {'table2': expected, 'displayed_cpu': actual,
                             'absolute_difference': None if expected is None or actual is None
                             else abs(float(actual) - float(expected))}
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('sample_id', 'fire_name', 'activity_bin', *PANEL_METRICS))
        writer.writeheader(); writer.writerow(row)
    return comparison


def _short_fire(name):
    return name.split('__')[0].replace('_', ' ').title().replace('Chimneytops2', 'Chimney Tops 2')


def draw_failure_figure(selection, data, output_dir, *, show_error_map=True, layout='compact',
                        formats=('pdf', 'svg', 'png'), filename='failure_case'):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    if layout not in ('compact', 'wide'):
        raise ValueError(f'Unknown layout: {layout}')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.3,
                         'axes.labelsize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    columns = 3 if show_error_map else 2
    width, height = (5.0, 6.5) if layout == 'compact' else (7.2, 6.5)
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(5, columns + 1, left=.17 if layout == 'compact' else .11,
                            right=.88 if layout == 'compact' else .90, top=.98, bottom=.035,
                            height_ratios=[.9, 1, 1, 1, 1], width_ratios=[*([1]*columns), .09],
                            hspace=.13, wspace=.12)
    context = data['context']
    observed_ax = fig.add_subplot(grid[0, 0])
    observed_ax.imshow(context['latest_observed_fire_state'], cmap='Greys', vmin=0, vmax=1,
                       interpolation='nearest', origin='upper')
    observed_ax.set_title('Observed $t$', pad=2)
    observed_ax.set_xticks([]); observed_ax.set_yticks([])
    u, v = context['wind_u'], context['wind_v']
    stride = 12
    yy, xx = np.mgrid[stride//2:u.shape[0]:stride, stride//2:u.shape[1]:stride]
    speed = np.hypot(u[yy, xx], v[yy, xx])
    uu = np.divide(u[yy, xx], speed, out=np.zeros_like(speed), where=speed > 1e-6)
    vv = np.divide(v[yy, xx], speed, out=np.zeros_like(speed), where=speed > 1e-6)
    observed_ax.quiver(xx, yy, uu, vv, color='#d5731a', angles='xy', scale_units='xy',
                       scale=.5, width=.003, pivot='middle', zorder=5)
    note_ax = fig.add_subplot(grid[0, 1:columns]); note_ax.axis('off')
    criterion = selection['criterion']
    value = selection['criterion_value']
    rank_label = selection.get('rank_label', 'test rank')
    if columns == 2:
        note = (f"{_short_fire(selection['fire_name'])}\n"
                f"Table 2: {criterion} · {rank_label} #{selection['test_rank']}\n"
                f"{DISPLAY_NAMES[criterion]}\n= {value:.3g}\n"
                f"$t$={selection['latest_input_timestamp']} → {selection['target_timestamp']}")
    else:
        note = (f"Fire: {_short_fire(selection['fire_name'])}\n"
                f"Table 2: {criterion} · {rank_label} {selection['test_rank']}\n"
                f"{DISPLAY_NAMES[criterion]} = {value:.3g}\n"
                f"Observed $t$={selection['latest_input_timestamp']} → future $t$={selection['target_timestamp']}")
    note_ax.text(.02, .92, note, va='top', ha='left', fontsize=7.3, transform=note_ax.transAxes)
    target = data['target']
    prediction = data['predictions']['final']
    limits = color_limits(target, 99)
    saved_limits = {}
    for row_index, (label, channel, cmap) in enumerate(ROWS, start=1):
        if channel == 2:
            vmin, vmax = 0.0, 1.0
            ground_truth = target[2]
            predicted = 1 / (1 + np.exp(-np.clip(prediction[2], -60, 60)))
        else:
            key = {0: 'surface', 1: 'canopy', 3: 'energy_log'}[channel]
            vmin, vmax = limits[key]
            ground_truth, predicted = target[channel], prediction[channel]
        arrays = [ground_truth, predicted]
        if show_error_map:
            arrays.append(np.abs(predicted - ground_truth))
        saved_limits[label] = [float(vmin), float(vmax)]
        for column, array in enumerate(arrays):
            ax = fig.add_subplot(grid[row_index, column])
            image = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax,
                              interpolation='nearest', origin='upper')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(.35); spine.set_color('#aaaaaa')
            if row_index == 1:
                ax.set_title(('GT', 'FLARE final', '|Prediction − GT|')[column], pad=2)
            if column == 0:
                ax.set_ylabel(label, labelpad=3)
        cax = fig.add_subplot(grid[row_index, columns])
        low = any(np.any(array < vmin) for array in arrays)
        high = any(np.any(array > vmax) for array in arrays)
        extend = 'both' if low and high else 'min' if low else 'max' if high else 'neither'
        bar = fig.colorbar(image, cax=cax, extend=extend)
        bar.ax.tick_params(labelsize=6, length=2, pad=1)
        if channel == 3:
            bar.ax.set_ylabel('log(1 + P_HRR)', fontsize=7, labelpad=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        if suffix not in ('pdf', 'svg', 'png'):
            raise ValueError(f'Unsupported figure format: {suffix}')
        fig.savefig(output_dir / f'{filename}.{suffix}', dpi=600)
    plt.close(fig)
    return saved_limits


def analyze_failure_cases(*, criterion='composite', top_k=50, selection_seeds=(42,), model_seed=42,
                          rank=None, allow_no_fire=False, show_error_map=True, layout='compact',
                          write_index=False):
    if criterion not in CRITERIA or model_seed not in (42, 123, 2026) or top_k < 1:
        raise ValueError('Invalid criterion, model seed, or top-k.')
    seeds = list(selection_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or rank is not None and len(seeds) != 1:
        raise ValueError('Provide distinct selection seeds; --rank requires one candidate.')
    torch.set_num_threads(2)
    dataset_config = load_config(DATASET_CONFIG)
    root, index, records, _ = read_test_records(dataset_config)
    by_id = {record['sample_id']: record for record in records}
    run = resolve_figure_run('final', model_seed)
    if run['signature'][-1] != len(records):
        raise RuntimeError('Final FLARE checkpoint has a different Table 2 test count.')
    metric_path = _metric_path(run, model_seed)
    try:
        rows = load_verified_metrics(metric_path, records, model_seed)
    except (RuntimeError, ValueError) as exc:
        print(f'Incompatible saved sample metrics ({exc}); recomputing the frozen CPU test table once.', flush=True)
        metric_path = _metric_path(run, model_seed, use_original=False, recompute=True)
        rows = load_verified_metrics(metric_path, records, model_seed)
    ranked = rank_failures(rows, criterion, allow_no_fire)
    candidate_root = _candidate_root(criterion, model_seed, top_k)
    _write_pool(candidate_root / 'ranked_pool.csv', ranked, top_k)
    selections = []
    cases = []
    for seed in seeds:
        row = choose_failure(ranked, top_k, seed, rank)
        case_dir = candidate_root / (f'rank_{rank:03d}' if rank is not None else f'seed_{seed:03d}')
        record = by_id[row['sample_id']]
        case = case_metadata(root, record)
        case.update({'case_type': 'selected', 'target_activity_bin': row['activity_bin'],
                     'target_active_fraction': row['active_fraction']})
        selection = {**row, 'criterion': criterion, 'top_k': top_k,
                     'selection_seed': None if rank is not None else seed, 'model_seed': model_seed,
                     'checkpoint': str(run['checkpoint']), 'metric_source': str(metric_path),
                     'split': 'test', 'selection_mode': 'exact_rank' if rank is not None else 'seeded_top_k',
                     'eligible_test_count': len(ranked), 'allow_no_fire': allow_no_fire,
                     'latest_input_timestamp': case['latest_input_timestamp'],
                     'target_timestamp': case['target_timestamp'],
                     'ranking_definition': ('mean of tie-aware percentile badness ranks for low Dice, '
                         'high HRR Log MAE, and high Active Canopy MAE' if criterion == 'composite'
                         else f'{"lowest" if criterion == "dice" else "highest"} {DISPLAY_NAMES[criterion]}')}
        path = case_dir / 'selected_sample.json'
        if path.is_file() and json.loads(path.read_text(encoding='utf-8')) != selection:
            raise RuntimeError(f'Existing failure selection differs: {path}')
        json_write(path, selection)  # durable objective selection before model inference
        cases.append(case)
        selections.append((selection, case_dir))
        print(f'SELECTED seed={seed}: rank={row["test_rank"]} {row["sample_id"]}', flush=True)
    if write_index:
        with (candidate_root / 'candidate_index.csv').open('w', newline='', encoding='utf-8') as handle:
            fields = ('selection_seed', 'sample_id', 'fire_name', 'activity_bin', 'criterion',
                      'criterion_value', 'rank_within_test', 'output_pdf')
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for selection, directory in selections:
                writer.writerow({'selection_seed': selection['selection_seed'],
                    'sample_id': selection['sample_id'], 'fire_name': selection['fire_name'],
                    'activity_bin': selection['activity_bin'], 'criterion': criterion,
                    'criterion_value': selection['criterion_value'],
                    'rank_within_test': selection['test_rank'],
                    'output_pdf': str(directory / 'failure_case.pdf')})
    data_list = _data_for_cases(root, cases, by_id, dataset_config)
    _load_one_model(run, cases, by_id, data_list, {})
    for (selection, directory), data in zip(selections, data_list):
        prediction = torch.from_numpy(data['predictions']['final'][None])
        target = torch.from_numpy(data['target'][None])
        computed = _sample_metric_rows(prediction, target,
            [{'sample_id': selection['sample_id'], 'fire_name': selection['fire_name']}],
            model_name='cawfe_latte_final', seed=model_seed)[0]
        metric_comparison = _save_panel_metrics(directory / 'panel_metrics.csv', selection, computed)
        _save_arrays(directory / 'predictions.npz', data)
        color = draw_failure_figure(selection, data, directory,
                                    show_error_map=show_error_map, layout=layout)
        json_write(directory / 'figure_metadata.json', {
            'schema_version': 1, 'selection_criterion': criterion,
            'criterion_value': selection['criterion_value'], 'test_rank': selection['test_rank'],
            'eligible_test_count': len(ranked), 'top_k': top_k,
            'ranked_pool': str(candidate_root / 'ranked_pool.csv'),
            'sample_metric_source': str(metric_path), 'sample_metric_source_sha256': sha256_file(metric_path),
            'checkpoint': str(run['checkpoint']), 'model_seed': model_seed,
            'layout': layout, 'show_error_map': show_error_map,
            'gt_only_color_limits': color, 'continuous_color_rule':
            '0 to 99th percentile of nonzero GT values; 1.0 fallback for empty GT',
            'mask_display': 'GT binary; FLARE sigmoid(logits); absolute probability error',
            'prediction_values_saved_raw': True,
            'table2_vs_displayed_cpu_metrics': metric_comparison,
        })
        print(f'FIGURE: {directory / "failure_case.pdf"}', flush=True)
    return candidate_root, selections


def _report(selection, directory):
    def show(value):
        return 'n/a' if value is None else f'{float(value):.6g}'
    print('FAILURE CASE ANALYSIS')
    print(f'FLARE checkpoint: {selection["checkpoint"]}')
    print(f'Model seed: {selection["model_seed"]}')
    print(f'Criterion: {selection["criterion"]}')
    print(f'Top-k: {selection["top_k"]}')
    print(f'Selection seed: {selection["selection_seed"] if selection["selection_seed"] is not None else "n/a (exact rank)"}')
    print(f'Selected sample: {selection["sample_id"]}')
    print(f'Fire: {selection["fire_name"]}')
    print(f'Test rank: {selection["test_rank"]} of {selection["eligible_test_count"]} eligible')
    print(f'Dice: {show(selection["dice"])}')
    print(f'HRR Log MAE: {show(selection["energy_log_mae"])}')
    print(f'Active Canopy MAE: {show(selection["active_canopy_mae"])}')
    print(f'Output: {directory / "failure_case.pdf"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--criterion', choices=CRITERIA, default='composite')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--model-seed', type=int, choices=(42, 123, 2026), default=42)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--allow-no-fire', action='store_true')
    parser.add_argument('--show-error-map', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--layout', choices=('compact', 'wide'), default='compact')
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.rank is not None and args.rank < 1:
        parser.error('--top-k and --rank must be positive')
    root, selections = analyze_failure_cases(criterion=args.criterion, top_k=args.top_k,
        selection_seeds=(args.selection_seed,), model_seed=args.model_seed, rank=args.rank,
        allow_no_fire=args.allow_no_fire, show_error_map=args.show_error_map, layout=args.layout)
    selection, directory = selections[0]
    _report(selection, directory)
    if args.rank is None:
        paper_figure = ROOT / 'figures/failure_case.pdf'
        paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / 'failure_case.pdf', paper_figure)
        print(f'PAPER FIGURE: {paper_figure}')
    print(f'OUTPUT DIRECTORY: {root}')
    return directory


if __name__ == '__main__':
    main()
