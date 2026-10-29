#!/usr/bin/env python3
"""Summarize completed validation-only FLARE terrain ablation runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path('artifacts/terrain_integration_ablation')
VARIANTS = (
    ('TERRAIN_NONE', 'No terrain'),
    ('TERRAIN_EARLY_CONCAT', 'Early concatenation'),
    ('TERRAIN_PREFUSION_FILM', 'Pre-fusion FiLM'),
    ('TERRAIN_QUERY', 'Terrain-conditioned query'),
    ('TERRAIN_LATE_CONCAT', 'Late concatenation'),
    ('TERRAIN_POSTFUSION_FILM', 'Post-fusion FiLM (current)'),
)
METRICS = {
    'dice': 'full_val_dice', 'iou': 'full_val_iou',
    'surface_mae': 'full_val_surface_mae', 'surface_rmse': 'full_val_surface_rmse',
    'canopy_mae': 'full_val_canopy_mae', 'canopy_rmse': 'full_val_canopy_rmse',
    'hrr_log_mae': 'full_val_energy_log_mae', 'hrr_log_rmse': 'full_val_energy_log_rmse',
    'active_canopy_mae': 'full_val_active_canopy_mae',
    'active_hrr_log_mae': 'full_val_active_energy_log_mae',
    'no_fire_mask_probability': 'full_val_no_fire_mask_prob_mean',
    'no_fire_pixel_fp_rate': 'full_val_no_fire_mask_false_positive_rate',
    'no_fire_patch_fp_rate': 'full_val_no_fire_patch_false_positive_rate',
    'no_fire_surface_abs_mean': 'full_val_no_fire_surface_abs_pred_mean',
    'no_fire_canopy_abs_mean': 'full_val_no_fire_canopy_abs_pred_mean',
    'no_fire_hrr_log_abs_mean': 'full_val_no_fire_energy_log_abs_pred_mean',
}
TABLE = ('dice', 'iou', 'surface_mae', 'canopy_mae', 'hrr_log_mae', 'active_canopy_mae')
HEADERS = ('Dice $\\uparrow$', 'IoU $\\uparrow$', 'Surf. MAE $\\downarrow$',
           'Canopy MAE $\\downarrow$', 'HRR Log MAE $\\downarrow$', 'Active Canopy MAE $\\downarrow$')


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def fmt(value: float | None) -> str:
    return '--' if value is None else f'{value:.4f}'


def main() -> None:
    analysis = ROOT / 'analysis'; analysis.mkdir(parents=True, exist_ok=True)
    runs, per_fire, regimes = [], [], []
    for variant, label in VARIANTS:
        for folder in sorted((ROOT / variant).glob('seed_*')):
            metadata_path = folder / 'terrain_ablation_metadata.json'
            metrics_path = folder / 'full_validation_metrics.json'
            if not metadata_path.is_file() or not metrics_path.is_file():
                continue
            metadata = read_json(metadata_path)
            metric_payload = read_json(metrics_path)
            metrics = metric_payload.get('metrics', metric_payload)
            if int(metadata.get('epochs', 0)) != 10 or metadata.get('test_evaluated') is not False:
                continue
            row = {'variant': variant, 'label': label, 'seed': metadata['seed'],
                   'best_epoch': metadata['best_epoch'], 'validation_samples': metadata['full_validation_samples'],
                   'total_parameters': metadata['parameter_counts']['total'],
                   'trainable_parameters': metadata['parameter_counts']['trainable']}
            for name, source in METRICS.items():
                row[name] = metrics.get(source)
            if any(row[name] is None for name in TABLE):
                continue
            runs.append(row)
            per_fire_path = folder / 'full_validation_per_fire.json'
            if per_fire_path.is_file():
                for fire, values in read_json(per_fire_path).get('fires', {}).items():
                    per_fire.append({'variant': variant, 'seed': metadata['seed'], 'fire_name': fire,
                                     **{('hrr_log' + key[len('energy_log'):]) if key.startswith('energy_log') else key: value for key, value in values.items()}})
            regime_path = folder / 'terrain_regime_metrics.json'
            if regime_path.is_file():
                payload = read_json(regime_path)
                for regime, values in payload.get('regimes', {}).items():
                    regimes.append({'variant': variant, 'seed': metadata['seed'], 'terrain_regime': regime,
                                    'slope_q33': payload['slope_quantile_boundaries'][0],
                                    'slope_q67': payload['slope_quantile_boundaries'][1],
                                    **{('hrr_log' + key[len('energy_log'):]) if key.startswith('energy_log') else key: value for key, value in values.items()}})
    write_csv(analysis / 'terrain_ablation_by_run.csv', runs,
              ['variant', 'label', 'seed', 'best_epoch', 'validation_samples', 'total_parameters', 'trainable_parameters', *METRICS])
    write_csv(analysis / 'terrain_ablation_per_fire.csv', per_fire,
              list(dict.fromkeys(k for row in per_fire for k in row)) or ['variant', 'seed', 'fire_name'])
    write_csv(analysis / 'terrain_ablation_by_terrain_regime.csv', regimes,
              list(dict.fromkeys(k for row in regimes for k in row)) or ['variant', 'seed', 'terrain_regime'])
    grouped = {}
    summary_rows = []
    for variant, label in VARIANTS:
        selected = [row for row in runs if row['variant'] == variant]
        if not selected:
            continue
        row = {'variant': variant, 'label': label, 'n_seeds': len(selected),
               'seeds': ' '.join(str(item['seed']) for item in selected)}
        for metric in METRICS:
            values = [float(item[metric]) for item in selected if item[metric] is not None]
            row[f'{metric}_mean'] = statistics.mean(values) if values else None
            row[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else None
        summary_rows.append(row); grouped[variant] = row
    write_csv(analysis / 'terrain_ablation_mean_std.csv', summary_rows,
              ['variant', 'label', 'n_seeds', 'seeds', *(key for metric in METRICS for key in (f'{metric}_mean', f'{metric}_std'))])
    complete_three = len(grouped) == len(VARIANTS) and all(row['n_seeds'] == 3 for row in summary_rows)
    if complete_three:
        caption_status = ''
    elif summary_rows and all(row['n_seeds'] == 1 for row in summary_rows):
        caption_status = ' Preliminary single-seed results; confirmation is pending.'
    else:
        caption_status = ' Preliminary results; fewer than three seeds are complete for at least one method.'
    best = {}
    for metric in TABLE:
        values = [row[f'{metric}_mean'] for row in summary_rows]
        if values:
            best[metric] = max(values) if metric in ('dice', 'iou') else min(values)
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{Effect of terrain integration on FLARE validation performance. All methods use GA\_Q2 and the same training protocol. Higher is better for Dice and IoU; lower is better for errors.' + caption_status + '}',
             r'\label{tab:terrain_integration}', r'\begin{tabular}{lcccccc}', r'\toprule',
             'Terrain integration & ' + ' & '.join(HEADERS) + r' \\', r'\midrule']
    for variant, label in VARIANTS:
        row = grouped.get(variant)
        cells = []
        for metric in TABLE:
            if row is None:
                cells.append('--'); continue
            value = row[f'{metric}_mean']; std = row[f'{metric}_std']
            cell = fmt(value) + (f' $\\pm$ {fmt(std)}' if std is not None else '')
            if len(grouped) == len(VARIANTS) and value is not None and math.isclose(value, best[metric], abs_tol=1e-12):
                cell = r'\textbf{' + cell + '}'
            cells.append(cell)
        lines.append(label.replace('_', r'\_') + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (analysis / 'terrain_ablation_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    text = ['TERRAIN INTEGRATION ABLATION', 'Base: FLARE final / GA_Q2; validation only; no test split constructed.',
            'Completed runs: ' + str(len(runs)), 'Three-seed confirmation complete: ' + str(complete_three),
            'Pairwise deltas use method minus reference; positive Dice/IoU and negative errors favor the method.']
    for reference in ('TERRAIN_POSTFUSION_FILM', 'TERRAIN_NONE'):
        text.append('\nRelative to ' + reference + ':')
        ref = grouped.get(reference)
        if ref is None:
            text.append('Reference unavailable.'); continue
        for variant, label in VARIANTS:
            row = grouped.get(variant)
            if row is None: continue
            text.append(label + ':')
            for metric in TABLE:
                base = float(ref[f'{metric}_mean']); value = float(row[f'{metric}_mean'])
                difference = value - base
                relative = 100 * difference / abs(base) if base else None
                text.append(f'  {metric}: {difference:+.6f}; relative {relative:+.2f}%' if relative is not None else f'  {metric}: {difference:+.6f}; relative undefined (zero reference)')
    def compare(method: str, reference: str) -> str:
        if method not in grouped or reference not in grouped:
            return f'{method} vs {reference}: pending runs.'
        better, worse, equal = [], [], []
        for metric in TABLE:
            delta = grouped[method][f'{metric}_mean'] - grouped[reference][f'{metric}_mean']
            directed = delta if metric in ('dice', 'iou') else -delta
            (better if directed > 1e-12 else worse if directed < -1e-12 else equal).append(metric)
        return (f'{method} vs {reference}: better in {", ".join(better) or "none"}; '
                f'worse in {", ".join(worse) or "none"}; tied in {", ".join(equal) or "none"}.')
    questions = ['Q1 Does terrain help?']
    questions += [compare(variant, 'TERRAIN_NONE') for variant, _ in VARIANTS if variant != 'TERRAIN_NONE']
    questions += ['Q2 Before or after attention?', compare('TERRAIN_PREFUSION_FILM', 'TERRAIN_POSTFUSION_FILM'),
                  compare('TERRAIN_QUERY', 'TERRAIN_POSTFUSION_FILM'),
                  'Q3 FiLM or concatenation?', compare('TERRAIN_PREFUSION_FILM', 'TERRAIN_EARLY_CONCAT'),
                  compare('TERRAIN_POSTFUSION_FILM', 'TERRAIN_LATE_CONCAT'),
                  'Q4 Does direct query conditioning help?', compare('TERRAIN_QUERY', 'TERRAIN_POSTFUSION_FILM'),
                  'Q5 Is the effect larger in complex terrain?']
    high = {(row['variant'], row['seed']): row for row in regimes if row['terrain_regime'] == 'high_slope'}
    low = {(row['variant'], row['seed']): row for row in regimes if row['terrain_regime'] == 'low_slope'}
    if not high or not low:
        questions.append('Pending complete validation terrain-regime results.')
    else:
        for variant, _ in VARIANTS:
            for regime, source in (('low_slope', low), ('high_slope', high)):
                values = [float(row['dice']) for (name, _), row in source.items() if name == variant and row.get('dice') is not None]
                if values:
                    questions.append(f'{variant} {regime}: mean per-sample Dice {statistics.mean(values):.4f} across {len(values)} runs; see by_terrain_regime.csv for errors.')
    text.append('\nScientific questions:')
    text.extend(questions)
    if not complete_three:
        text.append('Interpretation pending: complete three-seed results are required for a paper claim.')
    else:
        text.append('Interpret metric-specific deltas and terrain regimes; no composite score is used.')
    (analysis / 'terrain_ablation_summary.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(f'Wrote {analysis}; complete runs: {len(runs)}')


if __name__ == '__main__':
    main()
