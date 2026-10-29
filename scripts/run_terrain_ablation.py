#!/usr/bin/env python3
"""Train one validation-only GA_Q2 terrain-integration screening run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from src.config import load_config

MODES = ('none', 'early_concat', 'prefusion_film', 'terrain_query', 'late_concat', 'postfusion_film')
VARIANTS = {f'TERRAIN_{mode.upper()}': mode for mode in MODES}
VARIANTS['TERRAIN_QUERY'] = VARIANTS.pop('TERRAIN_TERRAIN_QUERY')
ROOT = Path('artifacts/terrain_integration_ablation')


def config_path(mode: str) -> Path:
    filename = 'query' if mode == 'terrain_query' else mode
    return Path('configs/terrain_ablation') / f'flare_terrain_{filename}.yaml'


def check_config(config: dict, mode: str, seed: int) -> None:
    training = config['training']
    assert config['model']['architecture'] == 'cawfe_latte'
    assert config['model']['terrain_integration']['mode'] == mode
    assert config['cawfe_latte']['fire_mmd']['enabled']
    assert int(training['max_epochs']) == int(training['epochs']) == 10
    assert int(training['max_train_batches']) == 7500
    assert int(training['batch_size']) == 8
    assert training.get('max_train_batches_per_epoch') is None
    assert training['validation']['screening']['sampling'] == 'stratified_fixed'
    assert training['validation']['full']['enabled'] and training['validation']['full']['checkpoint'] == 'best'
    assert not training['run_test_after_training'] and not training['run_external_test_after_training']
    assert config['dataloader']['include_test_split'] is False
    assert config['dataloader']['return_terrain'] is True
    assert config['evaluation']['split'] == 'val'
    assert int(training['seed']) == seed


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def complete_terrain_analysis(run_dir: Path) -> tuple[list[dict], list[float]]:
    sample_path = run_dir / 'evaluation/validation_sample_metrics.csv'
    with sample_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get('mean_slope') for row in rows):
        raise RuntimeError('Full validation sample metrics lack terrain statistics.')
    slopes = np.array([float(row['mean_slope']) for row in rows], dtype=np.float64)
    boundaries = np.quantile(slopes, [1/3, 2/3]).tolist()
    names = ('low_slope', 'medium_slope', 'high_slope')
    for row, slope in zip(rows, slopes):
        row['terrain_regime'] = names[int(slope > boundaries[0]) + int(slope > boundaries[1])]
    fields = list(rows[0]); temporary = sample_path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(sample_path)
    metrics = ('dice', 'surface_mae', 'canopy_mae', 'energy_log_mae')
    by_regime = {}
    for name in names:
        selected = [row for row in rows if row['terrain_regime'] == name]
        result = {'sample_count': len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) not in ('', None)]
            result[metric] = float(np.mean(values)) if values else None
        by_regime[name] = result
    write_json(run_dir / 'terrain_regime_metrics.json', {
        'metric_scope': 'complete_validation_best_checkpoint',
        'elevation_channel': 'relative_elevation', 'slope_channel': 'slope_magnitude',
        'slope_quantile_boundaries': boundaries,
        'aggregation': 'mean of per-sample metrics with defined values',
        'regimes': by_regime,
    })
    return rows, boundaries


def attention_diagnostics(config: dict, checkpoint: Path, run_dir: Path, boundaries: list[float]) -> None:
    import torch
    from src.data.dataset import create_dataloaders
    from src.models.model_factory import build_model_from_config
    from src.training.batch_utils import unpack_batch
    from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible
    from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, test_loader = create_dataloaders(config)
    if test_loader is not None:
        raise RuntimeError('Terrain ablation unexpectedly constructed a test loader.')
    model = build_model_from_config(config, input_channels=129).to(device)
    load_model_state_dict_compatible(model, load_checkpoint(checkpoint, map_location='cpu'), checkpoint)
    model.eval()
    normalizer = build_input_normalizer_for_loader(val_loader, device, 129, config)
    groups: dict[str, list[np.ndarray]] = {'all': [], 'low_slope': [], 'medium_slope': [], 'high_slope': []}
    # The first four sequential validation batches are fixed for every variant and seed.
    with torch.inference_mode():
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= 4:
                break
            x_raw, _, extra = unpack_batch(batch)
            terrain = extra['terrain']
            x = apply_input_normalization(x_raw.to(device), normalizer)
            features = model(x, terrain=terrain.to(device), return_features=True, return_attention=True)
            attention = features['fusion_attention'].float().mean(dim=(1, 2)).cpu().numpy()
            slope = terrain[:, 1].float().mean(dim=(-2, -1)).numpy()
            for weights, value in zip(attention, slope):
                regime = ('low_slope', 'medium_slope', 'high_slope')[int(value > boundaries[0]) + int(value > boundaries[1])]
                groups['all'].append(weights); groups[regime].append(weights)
    payload = {'metric_scope': 'first_four_sequential_validation_batches_best_checkpoint', 'modalities': ['atmosphere', 'wind', 'flux_hrr'], 'groups': {}}
    for name, values in groups.items():
        payload['groups'][name] = {'sample_count': len(values), 'mean_weights': np.stack(values).mean(axis=0).tolist() if values else None}
    write_json(run_dir / 'attention_diagnostics.json', payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('seed', type=int, choices=(42, 123, 2026))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--recover', action='store_true', help='Resume an incomplete checkpoint or finalize an existing complete validation run.')
    parser.add_argument('--finalize-only', action='store_true', help='Require completed training and full validation; never train.')
    args = parser.parse_args()
    mode = VARIANTS[args.variant]
    config = load_config(config_path(mode))
    run_dir = ROOT / args.variant / f'seed_{args.seed}'
    training = dict(config['training']); training['seed'] = args.seed
    training['run_name'] = f'seed_{args.seed}'; training['overwrite_run'] = False
    # RunManager appends run_name even with flat_run_layout=True.
    output = dict(training['output']); output['root_dir'] = str(ROOT / args.variant); output['flat_run_layout'] = True
    training['output'] = output; config['training'] = training; config['seed'] = args.seed
    check_config(config, mode, args.seed)
    if args.check_only:
        print(f'{args.variant} seed={args.seed}: config valid; output={run_dir}')
        return
    if args.finalize_only and not run_dir.exists():
        raise FileNotFoundError(f'No completed run to finalize: {run_dir}')
    if run_dir.exists() and not (args.recover or args.finalize_only):
        raise FileExistsError(f'Run directory already exists: {run_dir}; use --recover after checking it.')
    import torch
    from src.models.model_factory import build_model_from_config
    from src.training.train import train_model_from_config
    model = build_model_from_config(config, input_channels=129)
    model.alignment._spatial_position(64, 64, device=torch.device('cpu'), dtype=torch.float32)
    counts = {'total': sum(p.numel() for p in model.parameters()), 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    del model
    full_metrics_path = run_dir / 'full_validation_metrics.json'
    summary_path = run_dir / 'metadata/run_summary.json'
    if full_metrics_path.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        full_payload = json.loads(full_metrics_path.read_text(encoding='utf-8'))
        result = {
            'run_dir': str(run_dir),
            'history_rows': [None] * int(summary['epochs_completed']),
            'full_validation': {'metrics': full_payload['metrics']},
            'best_checkpoint_path': str(run_dir / 'checkpoints/best_model.pt'),
            'best_epoch': int(summary['best_epoch']),
        }
        print(f'Reusing completed training and full validation: {run_dir}', flush=True)
    else:
        if args.finalize_only:
            raise RuntimeError(f'Full validation is incomplete: {run_dir}')
        if run_dir.exists():
            latest = run_dir / 'checkpoints/latest_model.pt'
            if not latest.is_file():
                raise RuntimeError(f'Cannot safely resume without a latest checkpoint: {latest}')
            training['overwrite_run'] = True
            config['training'] = training
            checkpoint = dict(config.get('checkpoint', {}))
            checkpoint['resume'] = True
            config['checkpoint'] = checkpoint
            print(f'Resuming interrupted training from {latest}', flush=True)
        result = train_model_from_config(config)
    actual = Path(result['run_dir'])
    if actual.resolve() != run_dir.resolve():
        raise RuntimeError(f'Unexpected run directory: {actual}')
    if len(result['history_rows']) != 10:
        raise RuntimeError('Expected exactly 10 completed screening epochs.')
    full = result.get('full_validation', {})
    metrics = full.get('metrics', {})
    required = ('full_val_dice', 'full_val_iou', 'full_val_surface_mae', 'full_val_surface_rmse',
                'full_val_canopy_mae', 'full_val_canopy_rmse', 'full_val_energy_log_mae',
                'full_val_energy_log_rmse', 'full_val_active_canopy_mae', 'full_val_active_energy_log_mae',
                'full_val_no_fire_mask_prob_mean', 'full_val_no_fire_mask_false_positive_rate',
                'full_val_no_fire_patch_false_positive_rate', 'full_val_no_fire_surface_abs_pred_mean',
                'full_val_no_fire_canopy_abs_pred_mean', 'full_val_no_fire_energy_log_abs_pred_mean')
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f'Complete validation missing metrics: {missing}')
    best = Path(result['best_checkpoint_path'])
    if not best.is_file():
        raise RuntimeError(f'Missing best checkpoint: {best}')
    rows, boundaries = complete_terrain_analysis(actual)
    if mode in ('prefusion_film', 'terrain_query', 'postfusion_film'):
        attention_diagnostics(config, best, actual, boundaries)
    write_json(actual / 'terrain_ablation_metadata.json', {
        'variant': args.variant, 'mode': mode, 'seed': args.seed, 'epochs': 10,
        'training_batches_per_epoch': 7500, 'training_batch_size': 8,
        'training_sampling_protocol': 'epoch_random_subset_without_replacement_v1',
        'best_epoch': result['best_epoch'], 'best_checkpoint': str(best),
        'full_validation_samples': len(rows), 'parameter_counts': counts,
        'test_split_constructed': False, 'test_evaluated': False,
    })
    print(f'Complete validation: {actual}; best epoch {result["best_epoch"]}; Dice {metrics["full_val_dice"]}')


if __name__ == '__main__':
    main()
