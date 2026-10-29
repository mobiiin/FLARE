#!/usr/bin/env python3
"""Select held-out test no-fire/weak-fire cases and plot frozen FLARE models on CPU."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_qualitative_forecasts import case_metadata, raw_case
from scripts.plot_qualitative_test_comparison import (
    MODEL_KEYS, PAPER_NAMES, _data_for_cases, _load_one_model, _panel_metrics,
    json_write, resolve_figure_run,
)
from src.baselines.table2_deterministic import ProcessedHistoryBaselinePredictor
from src.config import load_config
from src.evaluation.fire_activity import active_fraction_bin_name, classify_fire_masks
from src.evaluation.qualitative_test_figure import gt_activity_index, read_test_records, sha256_file

OUTPUT_ROOT = ROOT / 'artifacts/background_weak_fire'
DATASET_CONFIG = ROOT / 'artifacts/table2_baselines/persistence/resolved_config.yaml'
ROW_SPECS = (('Active fire', 2), ('Surface fuel', 0), ('Canopy fuel', 1), ('HRR', 3))
MODEL_LABELS = ('GT', 'ConvLSTM U-Net', 'FLARE baseline', 'FLARE final')
METRIC_COLUMNS = ('sample_id', 'fire_name', 'model', 'Dice', 'IoU', 'Surface MAE',
                  'Canopy MAE', 'HRR Log MAE', 'Active Canopy MAE')


def select_pair(root, records, activity, dataset_config, seed, predictor=None):
    """Choose cases using only canonical GT classes and observed input history."""
    rng = random.Random(int(seed))
    ordered = sorted(records, key=lambda item: item['sample_id'])
    no_fire_pool = [r for r in ordered if activity[r['sample_id']]['activity_bin'] == 'no_fire'
                    and float(activity[r['sample_id']]['active_fraction']) == 0.0]
    weak_pool = [r for r in ordered if activity[r['sample_id']]['activity_bin'] in ('tiny_fire', 'small_fire')]
    if not no_fire_pool or not weak_pool:
        raise RuntimeError('Canonical held-out test no-fire or weak-fire pool is empty.')
    # A previous target at the forecast origin is already observed history.
    # Prefer no-fire patches whose footprint touches an active observed patch.
    active_at_origin = {}
    for record in ordered:
        if float(activity[record['sample_id']]['active_fraction']) > 0:
            active_at_origin.setdefault((record['fire_name'], record['target_index']), []).append(record)

    def touches(first, second):
        a, b = first['patch'], second['patch']
        return (a['x0'] <= b['x0'] + b['width'] and b['x0'] <= a['x0'] + a['width']
                and a['y0'] <= b['y0'] + b['height'] and b['y0'] <= a['y0'] + a['height'])

    nearby_pool = [record for record in no_fire_pool if any(
        touches(record, observed) for observed in active_at_origin.get(
            (record['fire_name'], record['current_index']), ()))]
    if not nearby_pool:
        raise RuntimeError('No no-fire test patch touches observed fire at its forecast origin.')
    rng.shuffle(nearby_pool)
    rng.shuffle(weak_pool)
    predictor = predictor or ProcessedHistoryBaselinePredictor('persistence', root, dataset_config)
    no_fire_record = nearby_pool[0]
    weak_record = next((r for r in weak_pool if r['fire_name'] != no_fire_record['fire_name']), weak_pool[0])
    selected = []
    for case_type, record in (('no_fire', no_fire_record), ('weak_fire', weak_record)):
        _, target, _ = raw_case(root, record, include_input=False, include_terrain=False)
        classified = classify_fire_masks(target[2][None])
        fraction = float(classified['active_fraction'][0])
        activity_bin = active_fraction_bin_name(fraction)
        row = activity[record['sample_id']]
        if (activity_bin != row['activity_bin'] or abs(fraction - float(row['active_fraction'])) > 1e-12
                or (case_type == 'no_fire' and fraction != 0)
                or (case_type == 'weak_fire' and activity_bin not in ('tiny_fire', 'small_fire'))):
            raise RuntimeError(f'Canonical GT activity mismatch for {record["sample_id"]}')
        case = case_metadata(root, record)
        case.update({'case_type': case_type, 'target_activity_bin': activity_bin,
                     'target_active_fraction': fraction,
                     'observed_active_fraction': float(np.mean(predictor.predict_one(record)[2] > 0)),
                     'nearby_observed_fire': case_type == 'no_fire'})
        selected.append(case)
    return selected


def attach_observed_full_masks(root, records, cases, case_data):
    """Show observed full-fire masks at each forecast origin, with patch boxes."""
    by_observed_time = {}
    for record in records:
        by_observed_time.setdefault((record['fire_name'], record['target_index']), record)
    for case, data in zip(cases, case_data):
        past = by_observed_time.get((case['fire_name'], case['latest_input_timestamp']))
        if past is None:
            continue
        with np.load(root / past['target_path'], allow_pickle=False) as archive:
            full = np.asarray(archive['fire_mask'] > 0.5, dtype=np.float32)
        patch = case['patch']
        y0, x0, height, width = (int(patch[key]) for key in ('y0', 'x0', 'height', 'width'))
        if full.shape[0] < y0 + height or full.shape[1] < x0 + width:
            raise RuntimeError(f'Observed full-fire mask misses {case["sample_id"]}')
        data['context']['observed_full_fire_state'] = full
        data['context']['observed_patch_bounds'] = (y0, x0, height, width)


def _figure_limits(case, data, channel):
    """One robust signed scale for every column in this case/continuous row.

    No-fire: 99th percentile of absolute predictions from the three models.
    Weak-fire: 99th percentile of absolute GT and predictions together.
    Both use symmetric limits, making negative predictions visible.
    """
    arrays = [np.asarray(data['predictions'][key][channel], dtype=np.float32) for key in MODEL_KEYS]
    if case['case_type'] != 'no_fire':
        arrays.insert(0, np.asarray(data['target'][channel], dtype=np.float32))
    values = np.concatenate([np.abs(array).ravel() for array in arrays])
    return max(float(np.percentile(values, 99)), 1e-6)


def draw_figure(cases, case_data, output_dir):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5,
                         'axes.labelsize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(7.4, 8.9))
    outer = fig.add_gridspec(2, 1, left=.074, right=.905, top=.985, bottom=.022, hspace=.095)
    limits = {}
    for case_index, (case, data) in enumerate(zip(cases, case_data)):
        block = outer[case_index].subgridspec(
            5, 6, height_ratios=[.30, 1, 1, 1, 1], width_ratios=[1.06, 1, 1, 1, 1, .09],
            hspace=.10, wspace=.13)
        title_ax = fig.add_subplot(block[0, :]); title_ax.axis('off')
        title = 'No future active fire' if case_index == 0 else 'Weak future active fire'
        fire = case['fire_name'].split('__')[0].replace('_', ' ').title().replace('Chimneytops2', 'Chimney Tops 2')
        title_ax.text(0, .55, f'({chr(97 + case_index)}) {title} · {fire}',
                      ha='left', va='center', fontsize=9, fontweight='semibold')
        context = data['context']
        observed_ax = fig.add_subplot(block[1, 0])
        observed = context.get('observed_full_fire_state', context['latest_observed_fire_state'])
        observed_ax.imshow(observed, cmap='Greys', vmin=0, vmax=1,
                           interpolation='nearest', origin='upper')
        observed_ax.set_title('Observed active fire', fontsize=8, pad=2)
        observed_ax.set_xticks([]); observed_ax.set_yticks([])
        bounds = context.get('observed_patch_bounds')
        if bounds is not None:
            from matplotlib.patches import Rectangle
            y0, x0, height, width = bounds
            observed_ax.add_patch(Rectangle((x0 - .5, y0 - .5), width, height, fill=False,
                                            edgecolor='#d5731a', linewidth=1.1, zorder=5))
        note_ax = fig.add_subplot(block[2:, 0]); note_ax.axis('off')
        context_note = 'Box: evaluated patch' if bounds is not None else 'Observed patch'
        note_ax.text(.02, .95, f"Observed $t$={case['latest_input_timestamp']}\n"
                     f"Future $t$={case['target_timestamp']}\n"
                     f"GT active {case['target_active_fraction']:.2%}\n"
                     f"Patch active {case['observed_active_fraction']:.2%}\n"
                     f"{context_note}", transform=note_ax.transAxes,
                     va='top', ha='left', fontsize=7, color='#444444')
        limits[case['sample_id']] = {}
        for row_index, (row_label, channel) in enumerate(ROW_SPECS, start=1):
            if channel == 2:
                vmin, vmax, cmap, norm = 0, 1, 'magma', None
                arrays = [data['target'][channel]] + [
                    1 / (1 + np.exp(-np.clip(data['predictions'][key][channel], -60, 60)))
                    for key in MODEL_KEYS]
            else:
                vmax = _figure_limits(case, data, channel)
                vmin, cmap, norm = -vmax, 'coolwarm', TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                arrays = [data['target'][channel]] + [data['predictions'][key][channel] for key in MODEL_KEYS]
            limits[case['sample_id']][row_label] = [float(vmin), float(vmax)]
            for column, array in enumerate(arrays, start=1):
                ax = fig.add_subplot(block[row_index, column])
                image = ax.imshow(array, cmap=cmap, vmin=vmin if norm is None else None,
                                  vmax=vmax if norm is None else None, norm=norm,
                                  interpolation='nearest', origin='upper')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(.35); spine.set_edgecolor('#aaaaaa')
                if row_index == 1:
                    ax.set_title(MODEL_LABELS[column-1], fontsize=8, pad=2)
                if column == 1:
                    ax.set_ylabel(row_label, labelpad=4)
            cax = fig.add_subplot(block[row_index, 5])
            bar = fig.colorbar(image, cax=cax, extend='both' if channel != 2 else 'neither')
            bar.ax.tick_params(labelsize=6, length=2, pad=1)
            if channel == 3:
                bar.ax.set_ylabel('log(1 + P_HRR)', fontsize=7, labelpad=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(output_dir / f'background_weak_fire_comparison.{suffix}', dpi=600)
    plt.close(fig)
    return limits


def _save_predictions(path, cases, case_data):
    arrays = {'channel_order': np.asarray(('surface_fuel_consumed', 'canopy_fuel_consumed',
                                          'active_fire_mask_or_logits', 'log1p_P_HRR'))}
    for label, case, data in zip(('a', 'b'), cases, case_data):
        arrays[f'case_{label}_sample_id'] = np.asarray(case['sample_id'])
        arrays[f'case_{label}_gt'] = np.asarray(data['target'], dtype=np.float32)
        for key in MODEL_KEYS:
            raw = np.asarray(data['predictions'][key], dtype=np.float32)
            arrays[f'case_{label}_{key}_raw'] = raw
            arrays[f'case_{label}_{key}_active_fire_probability'] = (
                1 / (1 + np.exp(-np.clip(raw[2], -60, 60)))).astype(np.float32)
    np.savez_compressed(path, **arrays)


def _save_metrics(path, cases, case_data):
    rows = []
    for row in _panel_metrics(cases, case_data):
        rows.append({'sample_id': row['sample_id'], 'fire_name': row['fire_name'],
                     'model': row['model'], 'Dice': row['dice'], 'IoU': row['iou'],
                     'Surface MAE': row['surface_mae'], 'Canopy MAE': row['canopy_mae'],
                     'HRR Log MAE': row['energy_log_mae'], 'Active Canopy MAE': row['active_canopy_mae']})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader(); writer.writerows(rows)


def generate_candidates(selection_seeds, model_seed=42, output_root=OUTPUT_ROOT, *, write_index=True):
    """Freeze all GT-only selections before loading any learned checkpoint."""
    if model_seed not in (42, 123, 2026):
        raise ValueError('model_seed must be 42, 123, or 2026')
    seeds = list(selection_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide one or more distinct selection seeds.')
    torch.set_num_threads(2)
    dataset_config = load_config(DATASET_CONFIG)
    root, index, records, _ = read_test_records(dataset_config)
    activity = gt_activity_index(records)
    predictor = ProcessedHistoryBaselinePredictor('persistence', root, dataset_config)
    by_id = {record['sample_id']: record for record in records}
    selections = []
    for seed in seeds:
        selected = select_pair(root, records, activity, dataset_config, seed, predictor)
        candidate_dir = Path(output_root) / f'seed_{seed:03d}'
        manifest_path = candidate_dir / 'selected_samples.json'
        manifest = {'schema_version': 1, 'selection_seed': seed,
                    'split': 'test', 'selection_uses_predictions': False,
                    'selection_rule': 'canonical no_fire touching an active patch at the observed forecast origin; canonical tiny_fire or small_fire',
                    'sample_index': str(index), 'sample_index_sha256': sha256_file(index),
                    'cases': selected}
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
            if existing != manifest:
                raise RuntimeError(f'Existing candidate selection differs: {manifest_path}')
        else:
            json_write(manifest_path, manifest)
        selections.append((seed, candidate_dir, selected))
        print(f'SELECTED seed={seed}: no-fire {selected[0]["sample_id"]} | '
              f'weak-fire {selected[1]["sample_id"]}', flush=True)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / 'candidate_index.csv'
    if write_index:
        with index_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=('selection_seed', 'no_fire_sample_id',
                'no_fire_fire_name', 'weak_fire_sample_id', 'weak_fire_fire_name', 'output_pdf'))
            writer.writeheader()
            for seed, candidate_dir, cases in selections:
                writer.writerow({'selection_seed': seed, 'no_fire_sample_id': cases[0]['sample_id'],
                                 'no_fire_fire_name': cases[0]['fire_name'],
                                 'weak_fire_sample_id': cases[1]['sample_id'],
                                 'weak_fire_fire_name': cases[1]['fire_name'],
                                 'output_pdf': str(candidate_dir / 'background_weak_fire_comparison.pdf')})
    cases = [case for _, _, pair in selections for case in pair]
    case_data = _data_for_cases(root, cases, by_id, dataset_config)
    attach_observed_full_masks(root, records, cases, case_data)
    runs = {}
    signature = None
    for key in MODEL_KEYS:
        run = resolve_figure_run(key, model_seed, signature)
        signature = run['signature']
        if signature[-1] != len(records):
            raise RuntimeError(f'Locked Table 2 test count differs for {key}')
        runs[key] = run
    reference_inputs = {}
    for key in MODEL_KEYS:
        print(f'MODEL {PAPER_NAMES[key]}: {runs[key]["checkpoint"]}', flush=True)
        _load_one_model(runs[key], cases, by_id, case_data, reference_inputs)
    for i, (seed, candidate_dir, pair) in enumerate(selections):
        data_pair = case_data[2*i:2*i+2]
        _save_predictions(candidate_dir / 'predictions.npz', pair, data_pair)
        _save_metrics(candidate_dir / 'panel_metrics.csv', pair, data_pair)
        limits = draw_figure(pair, data_pair, candidate_dir)
        json_write(candidate_dir / 'figure_metadata.json', {
            'schema_version': 1, 'selection_seed': seed, 'model_seed': model_seed,
            'selection_uses_predictions': False, 'continuous_color_rule':
            'symmetric ± 99th percentile absolute predictions for no-fire; symmetric ± 99th percentile absolute GT and predictions for weak-fire',
            'active_fire_color_limits': [0, 1], 'continuous_color_limits': limits,
            'checkpoints': {key: str(run['checkpoint']) for key, run in runs.items()},
            'selected_samples_sha256': sha256_file(candidate_dir / 'selected_samples.json'),
        })
        print(f'FIGURE seed={seed}: {candidate_dir / "background_weak_fire_comparison.pdf"}', flush=True)
    return output_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--model-seed', type=int, choices=(42, 123, 2026), default=42)
    args = parser.parse_args(argv)
    output = generate_candidates([args.selection_seed], args.model_seed, write_index=False)
    paper_figure = ROOT / 'figures/background_weak_fire_comparison.pdf'
    paper_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / f'seed_{args.selection_seed:03d}' / paper_figure.name, paper_figure)
    print(f'PAPER FIGURE: {paper_figure}')
    print(f'OUTPUT DIRECTORY: {output}')
    return output


if __name__ == '__main__':
    main()
