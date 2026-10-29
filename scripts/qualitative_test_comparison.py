#!/usr/bin/env python3
"""Save FLARE comparison PNGs for one fire, one split, or every fire.

Usage: python scripts/qualitative_test_comparison.py all 10
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import secrets

import numpy as np
import torch
import yaml

from scripts.export_qualitative_forecasts import case_metadata, raw_case, safe_name
from scripts.plot_qualitative_test_comparison import (
    MODEL_KEYS, ROOT, _data_for_cases, _load_one_model, resolve_figure_run,
)
from src.config import load_config
from src.evaluation.fire_activity import FIRE_MASK_THRESHOLD
from src.evaluation.qualitative_test_figure import draw_error_comparison_png, gt_descriptors

MODEL_SEED = 42
SPLITS = ('test', 'val', 'train')
CONFIG_PATH = ROOT / 'configs/plots/qualitative_test_comparison.yaml'
OUTPUT_DIR = ROOT / 'artifacts/qualitative_test_comparison'


def read_split_index(dataset_config):
    root = Path(dataset_config['dataloader']['dataset_root']).expanduser().resolve()
    index = root / 'indices/temporal' / f"samples_{dataset_config['dataloader']['sample_pattern']}.jsonl"
    if not index.is_file():
        raise FileNotFoundError(index)
    grouped = {split: defaultdict(list) for split in SPLITS}
    seen = set()
    with index.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split = record['split']
            if split not in grouped:
                raise ValueError(f'Unexpected dataset split {split!r} in {index}')
            sample_id = record['sample_id']
            if sample_id in seen:
                raise RuntimeError(f'Duplicate sample ID in temporal index: {sample_id}')
            seen.add(sample_id)
            grouped[split][record['fire_name']].append(record)
    metrics_path = ROOT / 'artifacts/table2_baselines/persistence/evaluation/test_metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    test_count = sum(map(len, grouped['test'].values()))
    if (metrics.get('metric_scope') != 'complete_locked_held_out_test'
            or metrics.get('split') != 'test' or test_count != int(metrics['dataset_sample_count'])):
        raise RuntimeError('Temporal test index differs from the locked Table 2 evaluation.')
    per_fire_path = ROOT / 'artifacts/table2_baselines/persistence/evaluation/test_per_fire_metrics.csv'
    with per_fire_path.open(newline='', encoding='utf-8') as handle:
        expected_fires = {row['fire_name'] for row in csv.DictReader(handle)}
    if set(grouped['test']) != expected_fires:
        raise RuntimeError('Temporal test fire set differs from the locked Table 2 evaluation.')
    counts = {split: sum(map(len, grouped[split].values())) for split in SPLITS}
    return root, grouped, counts


def select_fire_samples(root, records, count, *, active_check=None):
    """Choose distinct random timestamps and one fire-active patch at each."""
    if not records:
        raise ValueError('No samples are available for this fire and split.')
    if active_check is None:
        def active_check(record):
            _, target, _ = raw_case(root, record, include_input=False, include_terrain=False)
            return bool(np.any(target[2] > FIRE_MASK_THRESHOLD))
    by_timestamp = defaultdict(list)
    for record in records:
        by_timestamp[int(record['current_index'])].append(record)
    remaining = sorted(by_timestamp)
    split, fire = records[0]['split'], records[0]['fire_name']
    if count > len(remaining):
        raise ValueError(f'{split}/{fire} has {len(remaining)} distinct timestamps; requested {count} figures.')
    for timestamp in remaining:
        by_timestamp[timestamp].sort(key=lambda record: record['sample_id'])
    selected = []
    used_seeds = set()
    while len(selected) < count and remaining:
        seed = secrets.randbits(63)
        while seed in used_seeds:
            seed = secrets.randbits(63)
        used_seeds.add(seed)
        rng = random.Random(seed)
        timestamp = rng.choice(remaining)
        remaining.remove(timestamp)
        patches = by_timestamp[timestamp].copy()
        rng.shuffle(patches)
        record = next((record for record in patches if active_check(record)), None)
        if record is not None:
            selected.append((seed, record))
    if len(selected) != count:
        raise ValueError(f'{split}/{fire} has only {len(selected)} fire-active timestamps; requested {count} figures.')
    return selected


def selected_groups(grouped, selector):
    if selector.casefold() == 'all':
        return [(split, fire, records) for split in SPLITS
                for fire, records in sorted(grouped[split].items())]
    split_aliases = {'validation': 'val', 'training': 'train', 'trian': 'train'}
    normalized = split_aliases.get(selector.casefold(), selector.casefold())
    if normalized in SPLITS:
        split = normalized
        return [(split, fire, records) for fire, records in sorted(grouped[split].items())]
    matches = [(split, fire, records) for split in SPLITS
               for fire, records in grouped[split].items() if fire.casefold() == selector.casefold()]
    if len(matches) != 1:
        valid = ', '.join(sorted({fire for split in SPLITS for fire in grouped[split]}))
        raise ValueError(f'Choose all, test, val, train, or one fire name: {valid}')
    return matches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', help='Fire name, test, val, train, or all')
    parser.add_argument('figure_count', type=int, help='Number of PNGs per fire')
    args = parser.parse_args(argv)
    if args.figure_count < 1:
        parser.error('figure_count must be at least 1')
    torch.set_num_threads(2)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    config['layout']['num_cases'] = 1
    config['wind']['normalize_direction'] = True
    config['wind']['scale'] = 0.5
    dataset_config = load_config(ROOT / 'artifacts/table2_baselines/persistence/resolved_config.yaml')
    config['_dataset_config'] = dataset_config
    config['_wind_levels'] = dataset_config.get('atmospheric_features', {}).get('low_level_indices', [0, 1, 2])
    root, grouped, counts = read_split_index(dataset_config)
    try:
        groups = selected_groups(grouped, args.selection)
    except ValueError as exc:
        parser.error(str(exc))
    unit = 'PNG' if args.figure_count == 1 else 'PNGs'
    print(f'GROUPS: {len(groups)} fire/split groups; {args.figure_count} {unit} per group', flush=True)
    selections = []
    for split, fire, records in groups:
        for seed, record in select_fire_samples(root, records, args.figure_count):
            selections.append((split, fire, seed, record))
    del grouped, groups
    cases, by_id = [], {}
    for number, (split, fire, seed, record) in enumerate(selections, start=1):
        case = case_metadata(root, record)
        case.update(gt_descriptors(root, record))
        case.update({'case_type': 'selected', 'selection_seed': seed})
        cases.append(case)
        by_id[record['sample_id']] = record
        print(f'SELECTED {number}/{len(selections)}: {split}/{fire} | {case["sample_id"]} | '
              f'observed t={record["current_index"]} | selection seed={seed}', flush=True)
    case_data = _data_for_cases(root, cases, by_id, dataset_config)

    runs = {}
    reference_signature = None
    for key in MODEL_KEYS:
        run = resolve_figure_run(key, MODEL_SEED, reference_signature)
        reference_signature = run['signature']
        if reference_signature[-1] != counts['test']:
            raise RuntimeError(f'Table 2 test count differs from index for {key}')
        runs[key] = run
    reference_inputs = {}
    for key in MODEL_KEYS:
        print(f'MODEL: {key} | checkpoint={runs[key]["checkpoint"]}', flush=True)
        _load_one_model(runs[key], cases, by_id, case_data, reference_inputs, expected_counts=counts)

    outputs = []
    for number, ((split, fire, seed, record), case, data) in enumerate(zip(selections, cases, case_data), start=1):
        filename = f'{safe_name(fire)}_t{int(record["current_index"]):06d}_seed_{seed}_abs_error_full_fuel'
        output_dir = OUTPUT_DIR / split / safe_name(fire)
        output = output_dir / f'{filename}.png'
        draw_error_comparison_png(case, data, output, config)
        outputs.append(output)
        print(f'FIGURE {number}/{len(selections)}: {output}', flush=True)
    return outputs


if __name__ == '__main__':
    main()
