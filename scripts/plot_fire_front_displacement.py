#!/usr/bin/env python3
"""GT-selected burn-footprint boundary figure for frozen FLARE Table 2 checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from scripts.export_qualitative_forecasts import case_metadata, checked_case, raw_case
from scripts.plot_qualitative_test_comparison import (
    ROOT, _data_for_cases, _load_one_model, json_write, resolve_figure_run,
)
from src.config import load_config
from src.data.fire_mask_thresholds import resolve_frozen_thresholds
from src.evaluation.fire_activity import FIRE_MASK_THRESHOLD, PREDICTED_FIRE_THRESHOLD
from src.evaluation.fire_front_displacement import (
    boundary_metrics, draw_final, draw_preview, mask_descriptors, physical_context,
    reconstruct_current_mask, validate_future_target,
)
from src.evaluation.full_validation import FullValidationAccumulator
from src.evaluation.qualitative_test_figure import gt_activity_index, read_test_records, sha256_file

CASE_TYPES = ('directional', 'curved', 'difficult')
MODEL_KEYS = ('baseline', 'final')
DEFAULT_CONFIG = ROOT / 'configs/plots/fire_front_displacement.yaml'
DEFAULT_OUTPUT = ROOT / 'artifacts/fire_front_displacement'
TARGET_CONFIG = ROOT / 'configs/derived/default_with_fire_mask_thresholds.yaml'


def _target_protocol(root: Path):
    config = load_config(TARGET_CONFIG)
    thresholds, source = resolve_frozen_thresholds(config, TARGET_CONFIG, require=True)
    manifest_path = root / 'targets/h10/target_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if int(manifest['horizon']) != 10 or any(float(manifest['thresholds'][key]) != value for key, value in thresholds.items()):
        raise RuntimeError('Target manifest horizon or thresholds differ from the frozen target config.')
    return config, thresholds, source, manifest_path


def _selection_config(path: Path, args) -> dict:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if args.terrain is not None:
        config['terrain']['enabled'] = args.terrain == 'on'
    if args.wind is not None:
        config['wind']['enabled'] = args.wind == 'on'
    if int(config['figure']['dpi']) < 600:
        raise ValueError('Final PNG DPI must be at least 600.')
    return config


def _scan_candidates(root, records, dataset_config, target_config, thresholds, config, seed):
    activity = gt_activity_index(records)
    by_fire = {}
    selection = config['selection']
    for record in records:
        row = activity[record['sample_id']]
        pixels = int(row['active_pixel_count'])
        fraction = float(row['active_fraction'])
        if pixels >= int(selection['minimum_future_pixels']) and fraction <= float(selection['maximum_future_fraction']):
            by_fire.setdefault(record['fire_name'], []).append(record)
    rng = random.Random(seed)
    candidates = []
    pool_counts = {fire: len(pool) for fire, pool in sorted(by_fire.items())}
    for fire, pool in sorted(by_fire.items()):
        shortlist = rng.sample(pool, min(len(pool), int(selection['shortlist_per_fire'])))
        for record in shortlist:
            current = reconstruct_current_mask(root, record, target_config, thresholds)
            if int(current.sum()) < int(selection['minimum_current_pixels']):
                continue
            _, y, _ = raw_case(root, record, include_input=False, include_terrain=False)
            future = y[2] > FIRE_MASK_THRESHOLD
            context = physical_context(root, record, dataset_config)
            desc = mask_descriptors(current, future, context)
            if desc['centroid_displacement_pixels'] is None:
                continue
            candidates.append((record, desc))
    if len({record['fire_name'] for record, _ in candidates}) < 3:
        raise RuntimeError('GT-only shortlist has fewer than three distinct fires with reconstructable boundaries.')
    return candidates, pool_counts


def _ranks(candidates, key):
    values = np.asarray([desc[key] or 0.0 for _, desc in candidates], dtype=float)
    order = np.argsort(values, kind='stable')
    rank = np.empty(len(values), dtype=float)
    rank[order] = np.arange(len(values)) / max(1, len(values)-1)
    return rank


def _select_cases(candidates, pool_counts, root, index, seed, config):
    rng = random.Random(seed + 947)
    ranks = {key: _ranks(candidates, key) for key in (
        'centroid_displacement_pixels', 'dominant_component_fraction', 'boundary_complexity',
        'component_count', 'perimeter_pixels', 'terrain_elevation_std', 'terrain_slope_std',
        'wind_direction_variation')}
    selected, used_fires = [], set()
    for category in CASE_TYPES:
        ranked = []
        for i, (record, desc) in enumerate(candidates):
            if record['fire_name'] in used_fires:
                continue
            if category == 'directional':
                if (desc['current_border_margin_pixels']
                        < int(config['selection']['minimum_boundary_margin_pixels'])
                        or desc['future_border_margin_pixels']
                        < int(config['selection']['minimum_boundary_margin_pixels'])
                        or desc['dominant_component_fraction']
                        < float(config['selection']['minimum_dominant_component_fraction'])
                        or desc['centroid_displacement_pixels']
                        < float(config['selection']['minimum_centroid_displacement_pixels'])):
                    continue
                score = (4.0*desc['centroid_displacement_pixels']
                         + .5*ranks['dominant_component_fraction'][i]
                         - .05*ranks['boundary_complexity'][i])
            elif category == 'curved':
                score = (1.2*ranks['boundary_complexity'][i] + 1.0*ranks['component_count'][i]
                         + .4*ranks['perimeter_pixels'][i])
            else:
                score = (ranks['terrain_elevation_std'][i] + ranks['terrain_slope_std'][i]
                         + ranks['wind_direction_variation'][i] + .5*ranks['component_count'][i])
            ranked.append((float(score), record['sample_id'], record, desc))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            raise RuntimeError(f'No GT-only candidate remains for {category} on a distinct test fire.')
        limit = 1 if category == 'directional' else int(config['selection']['top_choice_count'])
        top = ranked[:min(len(ranked), limit)]
        score, _, record, desc = rng.choice(top)
        used_fires.add(record['fire_name'])
        case = case_metadata(root, record)
        case.update(desc)
        case.update({'case_type': category, 'selection_seed': seed,
                     'temporal_index': int(record['current_index']), 'crop_coordinates': record['patch'],
                     'selection_score_gt_only': score})
        selected.append(case)
    return {'schema_version': 1, 'selection_seed': seed, 'split': 'test',
            'selection_uses_predictions': False, 'selection_criteria': config['selection'],
            'candidate_pool_counts': pool_counts, 'sample_index_path': str(index),
            'sample_index_sha256': sha256_file(index), 'cases': selected}


def _validate_manifest(root, index, records, manifest, target_config, thresholds, dataset_config):
    if manifest.get('split') != 'test' or manifest.get('selection_uses_predictions') is not False:
        raise ValueError('Frozen selection must use GT/input only on the held-out test split.')
    if Path(manifest['sample_index_path']).resolve() != index.resolve() or manifest['sample_index_sha256'] != sha256_file(index):
        raise ValueError('Frozen selection index differs from the canonical test index.')
    cases = manifest['cases']
    if len(cases) != 3 or [case['case_type'] for case in cases] != list(CASE_TYPES):
        raise ValueError('Selection must have exactly directional, curved, and difficult cases.')
    if len({case['fire_name'] for case in cases}) != 3:
        raise ValueError('The three cases must come from different test fires.')
    by_id = {record['sample_id']: record for record in records}
    data = []
    for case in cases:
        record = by_id.get(case['sample_id'])
        if record is None or record['split'] != 'test':
            raise ValueError(f'Selected case is not in the test split: {case["sample_id"]}')
        checked_case(root, record, case)
        if int(record['current_index']) != int(case['temporal_index']) or record['patch'] != case['crop_coordinates']:
            raise ValueError(f'Selected temporal index or crop changed: {case["sample_id"]}')
        current = reconstruct_current_mask(root, record, target_config, thresholds)
        future = validate_future_target(root, record, target_config, thresholds)
        context = physical_context(root, record, dataset_config)
        desc = mask_descriptors(current, future, context)
        for field in desc:
            if field.startswith('centroid_') and field.endswith('_xy'):
                match = np.allclose(desc[field], case[field], rtol=0, atol=1e-9)
            else:
                match = np.isclose(float(desc[field]), float(case[field]), rtol=0, atol=1e-9)
            if not match:
                raise ValueError(f'Selected GT descriptor changed: {case["sample_id"]} {field}')
        data.append({'current_mask': current, 'future_mask': future, 'context': context})
    return cases, by_id, data


def _panel_metrics(cases, case_data, data):
    rows = []
    for case, item, source in zip(cases, case_data, data):
        target = torch.from_numpy(item['target'][None])
        for key in MODEL_KEYS:
            prediction = torch.from_numpy(item['predictions'][key][None])
            acc = FullValidationAccumulator()
            acc.update(prediction, target)
            result = acc.finalize()
            probability = 1/(1+np.exp(-np.clip(item['predictions'][key][2], -60, 60)))
            mask = probability > PREDICTED_FIRE_THRESHOLD
            row = {'sample_id': case['sample_id'], 'fire_name': case['fire_name'], 'case_type': case['case_type'],
                   'model_key': key, 'model': 'FLARE baseline' if key == 'baseline' else 'FLARE final',
                   'dice': result['full_val_dice'], 'iou': result['full_val_iou'],
                   'centroid_displacement_pixels': case['centroid_displacement_pixels'],
                   **boundary_metrics(source['future_mask'], mask)}
            rows.append(row)
    return rows


def _save_arrays(path, cases, data):
    arrays = {}
    for i, (case, item) in enumerate(zip(cases, data)):
        prefix = f'case_{i}_'
        arrays[prefix+'sample_id'] = np.asarray(case['sample_id'])
        arrays[prefix+'current_mask'] = item['current_mask'].astype(bool)
        arrays[prefix+'future_gt_mask'] = item['future_mask'].astype(bool)
        for name in ('wind_u', 'wind_v', 'terrain_elevation'):
            arrays[prefix+name] = np.asarray(item['context'][name], dtype=np.float32)
        for key in MODEL_KEYS:
            arrays[prefix+key+'_mask_probability'] = item['mask_probabilities'][key].astype(np.float32)
            arrays[prefix+key+'_predicted_mask'] = item['predicted_masks'][key].astype(bool)
    np.savez_compressed(path, **arrays)


def _load_arrays(path, cases):
    data = []
    with np.load(path, allow_pickle=False) as archive:
        for i, case in enumerate(cases):
            prefix = f'case_{i}_'
            if str(archive[prefix+'sample_id']) != case['sample_id']:
                raise ValueError('Saved arrays differ from frozen sample IDs.')
            data.append({'current_mask': archive[prefix+'current_mask'],
                         'future_mask': archive[prefix+'future_gt_mask'],
                         'context': {name: archive[prefix+name] for name in ('wind_u','wind_v','terrain_elevation')},
                         'mask_probabilities': {key: archive[prefix+key+'_mask_probability'] for key in MODEL_KEYS},
                         'predicted_masks': {key: archive[prefix+key+'_predicted_mask'] for key in MODEL_KEYS}})
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--model-seed', type=int, choices=(42, 123, 2026), default=42)
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--freeze-selection', type=Path)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--terrain', choices=('on','off'))
    parser.add_argument('--wind', choices=('on','off'))
    parser.add_argument('--show-centroid-arrow', action='store_true')
    parser.add_argument('--annotate-metrics', action='store_true')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.selection_only and args.render_only:
        parser.error('--selection-only and --render-only are mutually exclusive')
    torch.set_num_threads(args.cpu_threads)
    config = _selection_config(args.config, args)
    dataset_config = load_config(ROOT / 'artifacts/table2_baselines/persistence/resolved_config.yaml')
    root, index, records, _ = read_test_records(dataset_config)
    target_config, thresholds, threshold_source, target_manifest_path = _target_protocol(root)
    if args.freeze_selection:
        manifest = json.loads(args.freeze_selection.read_text(encoding='utf-8'))
        seed = int(manifest['selection_seed'])
    else:
        seed = int(args.selection_seed)
        stored = args.output_dir / f'selection_seed_{seed}' / 'selected_cases.json'
        if stored.is_file():
            manifest = json.loads(stored.read_text(encoding='utf-8'))
        elif args.render_only:
            raise FileNotFoundError(stored)
        else:
            candidates, pool_counts = _scan_candidates(root, records, dataset_config, target_config, thresholds, config, seed)
            manifest = _select_cases(candidates, pool_counts, root, index, seed, config)
    selection_dir = args.output_dir / f'selection_seed_{seed}'
    stored = selection_dir / 'selected_cases.json'
    if stored.is_file() and json.loads(stored.read_text(encoding='utf-8'))['cases'] != manifest['cases']:
        raise RuntimeError(f'Existing frozen selection differs from requested manifest: {stored}')
    cases, by_id, data = _validate_manifest(root, index, records, manifest, target_config, thresholds, dataset_config)
    if not stored.is_file():
        json_write(stored, manifest)  # GT-only validation finishes before any checkpoint is opened.
    preview = selection_dir / 'selection_preview.png'
    if not preview.is_file():
        draw_preview(cases, data, preview, config)
    if args.selection_only:
        print('GT/INPUT-ONLY SELECTION:', stored)
        print('GT/INPUT-ONLY PREVIEW:', preview)
        return selection_dir
    model_dir = selection_dir / f'model_seed_{args.model_seed}'
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        data = _load_arrays(model_dir / 'arrays.npz', cases)
        with (model_dir / 'panel_metrics.csv').open(newline='', encoding='utf-8') as handle:
            metrics = [{k: float(v) if k in ('dice','iou') else v for k,v in row.items()}
                       for row in csv.DictReader(handle)]
    else:
        runs = {}
        reference = None
        for key in MODEL_KEYS:
            run = resolve_figure_run(key, args.model_seed, reference)
            reference = run['signature']
            if reference[-1] != len(records):
                raise RuntimeError('Frozen run test count differs from the canonical index.')
            resolved_thresholds, _ = resolve_frozen_thresholds(run['config'], run['config_path'], require=True)
            if resolved_thresholds != thresholds:
                raise RuntimeError(f'Frozen {key} model thresholds differ from target construction.')
            runs[key] = run
        checkpoint_manifest = {'schema_version': 1, 'model_seed': args.model_seed,
            'checkpoint_selection': 'frozen Table 2 run_dirs; no test-based selection',
            'models': {key: {'paper_name': 'FLARE baseline' if key=='baseline' else 'FLARE final',
                'architecture': 'cawfe_latte', 'finalist': 'baseline' if key=='baseline' else 'GA_Q2',
                'checkpoint': str(run['checkpoint']), 'checkpoint_sha256': sha256_file(run['checkpoint']),
                'config': str(run['config_path']), 'normalization': str(run['normalization']),
                'locked_test_metrics': str(run['metrics_path'])} for key, run in runs.items()}}
        json_write(model_dir / 'checkpoint_manifest.json', checkpoint_manifest)
        case_data = _data_for_cases(root, cases, by_id, dataset_config)
        for item, loaded in zip(data, case_data):
            if not np.array_equal(item['future_mask'], loaded['target'][2] > FIRE_MASK_THRESHOLD):
                raise RuntimeError('Future GT mask differs from the model-loader target.')
        reference_inputs = {}
        for key in MODEL_KEYS:
            print('CHECKPOINT:', key, runs[key]['checkpoint'], flush=True)
            _load_one_model(runs[key], cases, by_id, case_data, reference_inputs)
        for item, loaded in zip(data, case_data):
            probabilities = {}; masks = {}
            for key in MODEL_KEYS:
                probabilities[key] = 1/(1+np.exp(-np.clip(loaded['predictions'][key][2], -60, 60)))
                masks[key] = probabilities[key] > PREDICTED_FIRE_THRESHOLD
            item['mask_probabilities'] = probabilities
            item['predicted_masks'] = masks
        metrics = _panel_metrics(cases, case_data, data)
        with (model_dir / 'panel_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader(); writer.writerows(metrics)
        _save_arrays(model_dir / 'arrays.npz', cases, data)
    draw_final(cases, data, model_dir, config, show_centroid_arrow=args.show_centroid_arrow,
               annotate_metrics=args.annotate_metrics, metrics=metrics)
    metadata = {'schema_version': 1, 'split': 'test', 'selection_seed': seed, 'model_seed': args.model_seed,
        'selected_cases': str(stored), 'selected_cases_sha256': sha256_file(stored),
        'sample_index': str(index), 'target_manifest': str(target_manifest_path),
        'target_semantics': 'interval burn-footprint masks, not instantaneous flame perimeters',
        'current_mask_construction': '[t-10,t] consumed surface/canopy fuel plus endpoint HRR at t',
        'future_mask_construction': '[t,t+10] canonical target channel 2',
        'target_builder': 'src.data.processed_targets.build_processed_target',
        'frozen_mask_thresholds': thresholds, 'threshold_source': threshold_source['source'],
        'target_mask_threshold': FIRE_MASK_THRESHOLD, 'prediction_probability_threshold': PREDICTED_FIRE_THRESHOLD,
        'wind_source': 'latest observed x_raw; configured low_level_indices and variables_per_level',
        'wind_display': 'direction-normalized arrows' if config['wind'].get('normalize_direction', False) else 'raw U/V vectors',
        'terrain_source': 'terrain_features.npy channel 0 (relative elevation)',
        'line_and_layout_config': config, 'selection_uses_predictions': False,
        'checkpoint_manifest': str(model_dir / 'checkpoint_manifest.json'),
        'boundary_distance_units': 'pixels/grid cells'}
    json_write(model_dir / 'figure_metadata.json', metadata)
    print('\nFIRE-FRONT / BURN-FOOTPRINT FIGURE')
    print('Selection mode: GT/input only')
    for label, case in zip(('A','B','C'), cases):
        print(f'  {label}: {case["case_type"]} | {case["fire_name"]} | {case["sample_id"]}')
    if not args.render_only:
        print('Baseline checkpoint:', runs['baseline']['checkpoint'])
        print('Final checkpoint:', runs['final']['checkpoint'])
    print('Model seed:', args.model_seed)
    print('Prediction threshold:', PREDICTED_FIRE_THRESHOLD)
    print('Current-mask construction: [t-10,t] interval burn footprint, exact canonical target builder')
    print('Wind source: latest observed raw frame, configured low-level U/V')
    print('Terrain source: relative elevation channel 0')
    print('Output:', model_dir / 'fire_front_displacement.pdf')
    print('CPU-only: YES')
    print('No model output used for case selection: YES')
    print('Terminology: interval burn-footprint boundaries; use "burn-footprint boundary" in the manuscript, not "fire perimeter".')
    return model_dir


if __name__ == '__main__':
    main()
