"""Ground-truth-only case selection and paper figure rendering helpers."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from src.baselines.table2_deterministic import ProcessedHistoryBaselinePredictor
from src.evaluation.fire_activity import FIRE_MASK_THRESHOLD, active_fraction_bin_name, classify_fire_masks
from scripts.export_qualitative_forecasts import case_metadata, raw_case

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ('medium', 'high_energy', 'weak')


def read_test_records(config):
    root = Path(config['dataloader']['dataset_root']).expanduser().resolve()
    index = root / 'indices/temporal' / f"samples_{config['dataloader']['sample_pattern']}.jsonl"
    if not index.is_file():
        raise FileNotFoundError(index)
    records = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get('split') == 'test']
    ids = [r['sample_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Test sample IDs are not unique.')
    metrics = json.loads((ROOT / 'artifacts/table2_baselines/persistence/evaluation/test_metrics.json').read_text())
    if (metrics.get('metric_scope') != 'complete_locked_held_out_test' or metrics.get('split') != 'test'
            or len(records) != int(metrics['dataset_sample_count'])):
        raise RuntimeError('Temporal index does not match the complete locked Table 2 test evaluation.')
    expected_fires = set()
    with (ROOT / 'artifacts/table2_baselines/persistence/evaluation/test_per_fire_metrics.csv').open(newline='') as handle:
        expected_fires = {r['fire_name'] for r in csv.DictReader(handle)}
    fires = sorted({r['fire_name'] for r in records})
    if set(fires) != expected_fires:
        raise RuntimeError(f'Test fire set differs from Table 2: {fires} versus {sorted(expected_fires)}')
    return root, index, records, fires


def gt_activity_index(records):
    """Read only target-derived fields; never read prediction or error columns."""
    path = ROOT / 'artifacts/table2_baselines/persistence/evaluation/test_sample_metrics.csv'
    allowed = {'sample_id', 'fire_name', 'activity_bin', 'active_fraction', 'active_pixel_count'}
    with path.open(newline='') as handle:
        rows = [{k: row[k] for k in allowed} for row in csv.DictReader(handle)]
    result = {r['sample_id']: r for r in rows}
    if len(result) != len(records) or set(result) != {r['sample_id'] for r in records}:
        raise RuntimeError('Canonical GT activity rows differ from test sample IDs.')
    return result


def perimeter(mask):
    active = np.asarray(mask, dtype=bool)
    neighbors = (np.pad(active[:-1], ((1, 0), (0, 0))) &
                 np.pad(active[1:], ((0, 1), (0, 0))) &
                 np.pad(active[:, :-1], ((0, 0), (1, 0))) &
                 np.pad(active[:, 1:], ((0, 0), (0, 1))))
    return int(np.count_nonzero(active & ~neighbors))


def gt_descriptors(root, record):
    _, target, _ = raw_case(root, record, include_input=False, include_terrain=False)
    mask = target[2] > FIRE_MASK_THRESHOLD
    state = classify_fire_masks(target[2][None])
    active_count = int(state['active_pixels'][0])
    fraction = float(state['active_fraction'][0])
    if active_count == 0:
        raise ValueError('Selected GT candidate unexpectedly has no fire.')
    canopy = target[1][mask]
    yy, xx = np.nonzero(mask)
    center_distance = float(np.hypot(yy.mean()/max(1,mask.shape[0]-1)-.5, xx.mean()/max(1,mask.shape[1]-1)-.5))
    interior = mask[8:-8,8:-8] if min(mask.shape) > 16 else mask
    return {
        'target_activity_bin': active_fraction_bin_name(fraction),
        'target_active_fraction': fraction,
        'target_active_pixels': active_count,
        'target_energy_log_mean': float(target[3].mean()),
        'target_energy_log_active_mean': float(target[3][mask].mean()),
        'target_surface_active_mean': float(target[0][mask].mean()),
        'target_canopy_active_mean': float(canopy.mean()),
        'target_canopy_active_fraction': float((canopy > 0).mean()),
        'target_interior_active_fraction': float(interior.sum()/active_count),
        'target_centroid_distance': center_distance,
        'target_perimeter_pixels': perimeter(mask),
        'target_boundary_complexity': float(perimeter(mask) / max(1, active_count)),
    }


def observed_context(root, record, config, predictor=None):
    """Build the latest *observed* interval mask and raw low-level wind."""
    predictor = predictor or ProcessedHistoryBaselinePredictor('persistence', root, config)
    observed_logits = predictor.predict_one(record)[2]
    observed_mask = (observed_logits > 0).astype(np.float32)
    patch = record['patch']
    y0, x0, h, w = (int(patch[k]) for k in ('y0', 'x0', 'height', 'width'))
    path = root / 'fires' / record['fire_name'] / 'frames' / f"frame_{int(record['current_index']):06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        frame = np.asarray(archive['x_raw'], dtype=np.float32)
    atmospheric = config.get('atmospheric_features', {})
    levels = [int(v) for v in atmospheric.get('low_level_indices', [0, 1, 2])]
    width = int(atmospheric.get('variables_per_level', 10))
    if not levels or width < 2 or max(levels) * width + 1 >= frame.shape[0]:
        raise ValueError(f'Invalid configured low-level U/V channels for {path}')
    u = np.mean(np.stack([frame[z*width, y0:y0+h, x0:x0+w] for z in levels]), axis=0).astype(np.float32)
    v = np.mean(np.stack([frame[z*width+1, y0:y0+h, x0:x0+w] for z in levels]), axis=0).astype(np.float32)
    surface_channel = int(config['multitask']['surface_fuel_channel'])
    if not 0 <= surface_channel < frame.shape[0]:
        raise ValueError(f'Configured surface-fuel channel {surface_channel} is invalid for {path}')
    surface_fuel_full = np.asarray(frame[surface_channel], dtype=np.float32)
    surface_fuel = np.asarray(surface_fuel_full[y0:y0+h, x0:x0+w], dtype=np.float32)
    if (surface_fuel.shape != (h, w) or not np.isfinite(surface_fuel_full).all()
            or not np.isfinite(surface_fuel).all()):
        raise ValueError(f'Invalid full-fire surface-fuel frame or patch in {path}')
    terrain_path = root / 'fires' / record['fire_name'] / 'terrain/terrain_features.npy'
    elevation = None
    if terrain_path.is_file():
        terrain = np.load(terrain_path, allow_pickle=False)
        elevation = np.asarray(terrain[0, y0:y0+h, x0:x0+w], dtype=np.float32)
    return {'latest_observed_fire_state': observed_mask,
            'surface_fuel': surface_fuel, 'surface_fuel_full': surface_fuel_full,
            'surface_fuel_patch_bounds': {'y0': y0, 'x0': x0, 'height': h, 'width': w},
            'surface_fuel_channel': surface_channel,
            'wind_u': u, 'wind_v': v, 'terrain_elevation': elevation}


def _base_pool(records, activity, selection):
    pools = defaultdict(list)
    counts = Counter()
    for record in records:
        row = activity[record['sample_id']]
        name = row['activity_bin']
        pixels = int(row['active_pixel_count'])
        category = None
        if name == 'medium_fire' and pixels >= int(selection['medium_min_pixels']):
            category = 'medium'
        elif name == 'large_fire' and pixels >= int(selection['large_min_pixels']):
            category = 'high_energy'
        elif name in ('small_fire', 'tiny_fire') and pixels >= int(selection['weak_min_pixels']):
            category = 'weak'
        if category:
            pools[(category, record['fire_name'])].append(record)
            counts[category] += 1
    return pools, counts


def select_gt_cases(root, index, records, config, seed, num_cases):
    selection = config['selection']
    activity = gt_activity_index(records)
    pools, initial_counts = _base_pool(records, activity, selection)
    categories = CATEGORIES[:num_cases]
    predictor = ProcessedHistoryBaselinePredictor('persistence', root, config['_dataset_config'])
    rng = random.Random(int(seed))
    inspected = defaultdict(list)
    for category in categories:
        for fire in sorted({r['fire_name'] for r in records}):
            pool = pools.get((category, fire), [])
            if not pool:
                continue
            # A fixed-size GT-only shortlist bounds target I/O. Sampling is
            # seeded before any model checkpoint is opened.
            shortlist = rng.sample(pool, min(len(pool), int(selection['shortlist_per_fire'])))
            for record in shortlist:
                desc = gt_descriptors(root, record)
                if desc['target_centroid_distance'] > float(selection['max_centroid_distance']):
                    continue
                if category == 'medium' and (desc['target_perimeter_pixels'] < int(selection['medium_min_perimeter'])
                    or desc['target_interior_active_fraction'] < float(selection['medium_min_interior_fraction'])):
                    continue
                if category == 'high_energy' and (desc['target_energy_log_active_mean'] <= 0
                    or desc['target_interior_active_fraction'] < float(selection['large_min_interior_fraction'])):
                    continue
                inspected[category].append((record, desc, rng.random()))
    selected, used_fires = [], set()
    summary = {'schema_version': 1, 'selection_seed': int(seed), 'source': 'target-derived Table 2 sample activity and processed GT arrays',
               'total_test_records': len(records), 'test_fires': sorted({r['fire_name'] for r in records}),
               'initial_pool_counts': dict(initial_counts),
               'inspected_gt_counts': {category: len(inspected[category]) for category in categories},
               'selection_criteria': dict(selection)}
    for category in categories:
        available = [(r, d, tie) for r, d, tie in inspected[category] if r['fire_name'] not in used_fires]
        if not available:
            raise RuntimeError(f'No GT-only {category} candidate remains on a distinct test fire.')
        if category == 'high_energy':
            cutoff = float(np.quantile([d['target_energy_log_active_mean'] for _, d, _ in available],
                                       float(selection['high_energy_quantile'])))
            available = [(r, d, tie) for r, d, tie in available if d['target_energy_log_active_mean'] >= cutoff]
            summary['high_energy_active_log_quantile_cutoff'] = cutoff
        if category == 'medium':
            available.sort(key=lambda x: (-(x[1]['target_perimeter_pixels'] + 400*x[1]['target_active_fraction']
                                                   + 80*x[1]['target_interior_active_fraction'] - 80*x[1]['target_centroid_distance']), x[2]))
        elif category == 'high_energy':
            available.sort(key=lambda x: (-(x[1]['target_energy_log_active_mean']
                + 1.5*x[1]['target_interior_active_fraction'] + 0.5*x[1]['target_canopy_active_fraction']
                - x[1]['target_centroid_distance']), x[2]))
        else:
            available.sort(key=lambda x: (-(x[1]['target_perimeter_pixels']), x[2]))
        top = available[:min(len(available), int(selection['top_choice_count']))]
        # Prefer an observed-to-future change, using observed frames and GT only.
        enriched = []
        for record, desc, tie in top:
            _, target, _ = raw_case(root, record, include_input=False, include_terrain=False)
            context = observed_context(root, record, config['_dataset_config'], predictor)
            future = target[2] > FIRE_MASK_THRESHOLD
            observed = context['latest_observed_fire_state'] > 0.5
            displacement = float(np.mean(future != observed))
            enriched.append((record, desc, tie, displacement))
        if category == 'high_energy':
            enriched.sort(key=lambda x: (-(x[1]['target_energy_log_active_mean']
                + 1.5*x[1]['target_interior_active_fraction'] + .5*x[3]), x[2]))
        else:
            enriched.sort(key=lambda x: (-x[3], -x[1]['target_perimeter_pixels'], x[2]))
        # Seeded tie choice among top three GT/context candidates creates
        # representative variety without consulting any prediction.
        choice = enriched[rng.randrange(min(3, len(enriched)))]
        record, desc, tie, displacement = choice
        used_fires.add(record['fire_name'])
        canonical = case_metadata(root, record)
        canonical.update(desc)
        canonical.update({'case_type': category, 'selection_seed': int(seed),
                          'observed_future_activity_displacement': displacement,
                          'selection_tiebreak': tie,
                          'selection_criteria': {'category': category, 'activity_bin': desc['target_activity_bin'],
                                                 'minimum_pixels': selection[f'{category}_min_pixels'] if category != 'high_energy' else selection['large_min_pixels'],
                                                 'different_fire': True}})
        selected.append(canonical)
        print(f"SELECTED {category}: {record['sample_id']} | fire={record['fire_name']} | patch={record['patch']} | t={record['current_index']} -> {record['target_index']}")
    return {'schema_version': 1, 'selection_seed': int(seed), 'split': 'test', 'sample_index_path': str(index),
            'sample_index_sha256': sha256_file(index), 'selection_uses_predictions': False,
            'cases': selected}, summary


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def color_limits(target, percentile, show_negative=False):
    limits = {'mask': (0.0, 1.0)}
    for name, channel in (('surface', 0), ('canopy', 1), ('energy_log', 3)):
        values = np.asarray(target[channel], dtype=np.float32)
        nonzero = values[values > 0]
        vmax = float(np.percentile(nonzero, percentile)) if nonzero.size else float(max(1.0, values.max()))
        vmax = max(vmax, 1e-6)
        limits[name] = (-vmax if show_negative else 0.0, vmax)
    return limits



def absolute_error_maps(target, predictions, model_keys, channel):
    """Return un-clipped |prediction - GT| for one continuous target channel."""
    if channel not in (0, 1, 3):
        raise ValueError('Absolute-error maps apply only to surface, canopy, and energy log.')
    truth = np.asarray(target[channel], dtype=np.float32)
    result = {}
    for key in model_keys:
        predicted = np.asarray(predictions[key][channel], dtype=np.float32)
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError(f'Invalid {key} prediction for channel {channel}')
        result[key] = np.abs(predicted - truth)
    return result


def absolute_error_limit(errors, percentile):
    """One robust color limit shared by all models in a channel row."""
    values = np.concatenate([np.ravel(array) for array in errors.values()])
    positive = values[values > 0]
    return max(float(np.percentile(positive, percentile)), 1e-6) if positive.size else 1e-6


def _style(config):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    fonts = config['fonts']
    plt.rcParams.update({
        'font.family': fonts['family'], 'font.size': fonts['base_size'],
        'axes.titlesize': fonts['title_size'], 'axes.labelsize': fonts['label_size'],
        'xtick.labelsize': fonts['tick_size'], 'ytick.labelsize': fonts['tick_size'],
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': config['figure']['background'],
        'savefig.facecolor': config['figure']['background'],
    })
    return plt


def _image_axis(ax, array, *, cmap, vmin, vmax):
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', interpolation='nearest', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.35)
        spine.set_color('#aaaaaa')
    return im


def _draw_context(ax, context, config):
    mask = context['latest_observed_fire_state']
    _image_axis(ax, mask, cmap='Greys', vmin=0, vmax=1)
    terrain = config['terrain']
    elevation = context.get('terrain_elevation')
    if terrain['contours'] and elevation is not None and float(np.max(elevation) - np.min(elevation)) > 1e-6:
        levels = np.linspace(float(np.min(elevation)), float(np.max(elevation)), int(terrain['contour_levels'])+2)[1:-1]
        ax.contour(elevation, levels=levels, colors='#4a696c', linewidths=float(terrain['linewidth']),
                   alpha=float(terrain['alpha']))
    wind = config['wind']
    if wind['enabled']:
        stride = int(wind['stride'])
        if stride < 1:
            raise ValueError('wind.stride must be positive')
        u, v = context['wind_u'], context['wind_v']
        yy, xx = np.mgrid[stride//2:u.shape[0]:stride, stride//2:u.shape[1]:stride]
        u_plot, v_plot = u[yy, xx].copy(), v[yy, xx].copy()
        if wind.get('normalize_direction', False):
            speed = np.hypot(u_plot, v_plot)
            u_plot = np.divide(u_plot, speed, out=np.zeros_like(u_plot), where=speed > 1e-6)
            v_plot = np.divide(v_plot, speed, out=np.zeros_like(v_plot), where=speed > 1e-6)
        ax.quiver(xx, yy, u_plot, v_plot, color='#d5731a', angles='xy', scale_units='xy',
                  scale=float(wind['scale']), width=float(wind['width']), pivot='middle', zorder=5)
    ax.set_title('Observed $t$', pad=2)


def draw_selection_sheet(cases, case_data, path, config):
    """A model-free review sheet generated before checkpoint loading."""
    plt = _style(config)
    width = float(config['figure']['width'])
    fig, axes = plt.subplots(len(cases), 5, figsize=(width, 1.75*len(cases)+0.55), squeeze=False)
    fig.subplots_adjust(left=.055, right=.99, top=.85, bottom=.14, wspace=.12, hspace=.76)
    labels = ('Observed + wind', 'Future GT mask', 'GT surface', 'GT canopy', 'GT energy log')
    for i, (case, data) in enumerate(zip(cases, case_data)):
        target, context = data['target'], data['context']
        limits = color_limits(target, config['continuous']['color_percentile'])
        _draw_context(axes[i,0], context, config)
        for j, (channel, name, cmap) in enumerate(((2,'mask',config['mask']['cmap']),
            (0,'surface',config['continuous']['cmaps']['surface']),
            (1,'canopy',config['continuous']['cmaps']['canopy']),
            (3,'energy_log',config['continuous']['cmaps']['energy_log'])), start=1):
            _image_axis(axes[i,j], target[channel], cmap=cmap, vmin=limits[name][0], vmax=limits[name][1])
        for ax, label in zip(axes[i], labels):
            ax.set_title(label, fontsize=config['fonts']['label_size'], pad=2)
        axes[i,0].text(0, -0.18, f"{case['fire_name']} · {case['target_activity_bin']} · "
            f"active {case['target_active_fraction']:.1%} · energy {case['target_energy_log_active_mean']:.2f} · "
            f"canopy {case['target_canopy_active_mean']:.2f}",
            transform=axes[i,0].transAxes, fontsize=config['fonts']['tick_size'], ha='left', va='top', clip_on=False)
    fig.suptitle('Held-out test candidates · ground truth and observed context only', fontsize=config['fonts']['title_size'])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(config['figure']['dpi']))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def _short_fire_name(name):
    base = name.split('__')[0].replace('_',' ').title()
    return base.replace('Chimneytops2','Chimney Tops 2')


def draw_comparison(cases, case_data, output_dir, config, *, annotate_metrics=False, panel_metrics=None,
                    formats=('pdf', 'svg', 'png'), filename='qualitative_test_comparison'):
    """Draw aligned GT/model panels with GT-only limits and vector labels."""
    plt = _style(config)
    layout = config['layout']['mode']
    if layout not in ('compact', 'stacked'):
        raise ValueError(f'Unknown layout {layout!r}')
    width = float(config['figure']['width'])
    height = float(config['layout']['case_height']) * len(cases) + .18
    fig = plt.figure(figsize=(width, height))
    outer = fig.add_gridspec(len(cases), 1, hspace=float(config['layout']['hspace']),
                             left=.075, right=.91, top=.985, bottom=.025)
    row_specs = [('mask',2,'Fire prob.'), ('surface',0,'Surface'), ('canopy',1,'Canopy'), ('energy_log',3,'Energy log')]
    model_keys = ('convlstm', 'baseline', 'final')
    display_names = ('GT', 'ConvLSTM', 'FLARE baseline', 'FLARE final')
    saved_limits = {}
    metric_index = {(r['sample_id'],r['model_key']): r for r in (panel_metrics or [])}
    for case_index, (case, data) in enumerate(zip(cases, case_data)):
        target, context, predictions = data['target'], data['context'], data['predictions']
        limits = color_limits(target, float(config['continuous']['color_percentile']),
                              bool(config['continuous'].get('show_negative', False)))
        saved_limits[case['sample_id']] = {key: list(value) for key, value in limits.items()}
        if layout == 'compact':
            block = outer[case_index].subgridspec(5,6, height_ratios=[.25,1,1,1,1],
                width_ratios=[float(config['layout']['context_width_ratio']),1,1,1,1,.085],
                hspace=float(config['layout']['hspace']), wspace=float(config['layout']['wspace']))
            title_ax = fig.add_subplot(block[0,:]); title_ax.axis('off')
            context_ax = fig.add_subplot(block[1,0])
            note_ax = fig.add_subplot(block[2:5,0]); note_ax.axis('off')
            image_row_offset, image_col_offset, cbar_col = 1, 1, 5
        else:
            block = outer[case_index].subgridspec(6,5, height_ratios=[.25,.82,1,1,1,1],
                width_ratios=[1,1,1,1,.085], hspace=float(config['layout']['hspace']),
                wspace=float(config['layout']['wspace']))
            title_ax = fig.add_subplot(block[0,:]); title_ax.axis('off')
            context_ax = fig.add_subplot(block[1,0])
            note_ax = fig.add_subplot(block[1,1:4]); note_ax.axis('off')
            image_row_offset, image_col_offset, cbar_col = 2, 0, 4
        names = {'medium':'Medium activity', 'high_energy':'High energy', 'weak':'Weak fire'}
        if case['case_type'] == 'selected':
            split_name = {'train':'training', 'val':'validation', 'test':'test'}[case['split']]
            title = f'Selected {split_name} sample'
        else:
            title = names[case['case_type']]
        title_ax.text(0, .5, f"({chr(97+case_index)}) {title} · {_short_fire_name(case['fire_name'])}",
                      ha='left', va='center', fontsize=config['fonts']['title_size'], fontweight='semibold')
        _draw_context(context_ax, context, config)
        note_ax.text(.03, .96, f"Observed $t$={case['latest_input_timestamp']}\nFuture $t$={case['target_timestamp']}\n"
            f"GT active {case['target_active_fraction']:.1%}\nWind: lowest {len(config['_wind_levels'])} levels",
            transform=note_ax.transAxes, va='top', ha='left', fontsize=config['fonts']['tick_size'], color='#444444')
        gt_mask = target[2] > FIRE_MASK_THRESHOLD
        for row_index, (quantity, channel, row_label) in enumerate(row_specs):
            vmin,vmax = limits[quantity]
            cmap = config['mask']['cmap'] if quantity == 'mask' else config['continuous']['cmaps'][quantity]
            images = [target[channel]]
            for model_key in model_keys:
                pred = predictions[model_key]
                if quantity == 'mask':
                    values = 1/(1+np.exp(-np.clip(pred[2],-60,60)))
                else:
                    values = pred[channel]
                    if config['continuous']['clip_negative_for_display']:
                        values = np.maximum(values,0)
                images.append(values)
            exceed_hi = any(np.any(values > vmax) for values in images[1:])
            exceed_lo = any(np.any(values < vmin) for values in images[1:])
            for column, values in enumerate(images):
                ax = fig.add_subplot(block[image_row_offset+row_index,image_col_offset+column])
                im = _image_axis(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
                if row_index == 0:
                    ax.set_title(display_names[column], pad=2)
                if column == 0:
                    ax.set_ylabel(row_label, labelpad=3)
                if quantity == 'mask' and column > 0 and config['mask']['show_gt_perimeter'] and gt_mask.any() and not gt_mask.all():
                    ax.contour(gt_mask.astype(float), levels=[.5], colors='white', linewidths=float(config['mask']['perimeter_linewidth']))
                if quantity == 'mask' and column > 0 and config['mask']['show_threshold_contour']:
                    if np.any(values < .5) and np.any(values > .5):
                        ax.contour(values, levels=[.5], colors='#44e1d5', linewidths=.5)
                if annotate_metrics and column > 0:
                    row = metric_index[(case['sample_id'],model_keys[column-1])]
                    value = row['dice'] if quantity == 'mask' else row[f'{quantity}_mae']
                    label = ('Dice ' if quantity == 'mask' else 'MAE ') + (f'{float(value):.2f}' if value not in ('',None) else 'n/a')
                    ax.text(.98,.03,label,ha='right',va='bottom',transform=ax.transAxes,fontsize=5.3,
                            color='white',bbox={'facecolor':'black','edgecolor':'none','alpha':.5,'pad':1})
            cax = fig.add_subplot(block[image_row_offset+row_index,cbar_col])
            extend = 'both' if exceed_hi and exceed_lo else 'max' if exceed_hi else 'min' if exceed_lo else 'neither'
            colorbar = fig.colorbar(im, cax=cax, extend=extend)
            colorbar.ax.tick_params(labelsize=config['fonts']['colorbar_size'], length=2, pad=1)
            if quantity == 'energy_log':
                colorbar.ax.set_ylabel('log(1+MW)', fontsize=config['fonts']['colorbar_size'], labelpad=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        if extension not in ('pdf', 'svg', 'png'):
            raise ValueError(f'Unsupported figure format: {extension}')
        fig.savefig(output_dir / f'{filename}.{extension}', dpi=int(config['figure']['dpi']))
    plt.close(fig)
    return saved_limits


def draw_error_comparison_png(case, data, output, config):
    """One-sample PNG: GT fields beside absolute model errors, plus observed fuel."""
    plt = _style(config)
    if config['layout']['mode'] != 'compact':
        raise ValueError('The per-fire absolute-error figure requires the compact layout.')
    target, context, predictions = data['target'], data['context'], data['predictions']
    model_keys = ('convlstm', 'baseline', 'final')
    display_names = ('GT', 'ConvLSTM', 'FLARE baseline', 'FLARE final')
    limits = color_limits(target, float(config['continuous']['color_percentile']))
    width = float(config['figure']['width'])
    height = float(config['layout']['case_height']) + .18
    fig = plt.figure(figsize=(width, height))
    block = fig.add_gridspec(5, 7, left=.075, right=.91, top=.985, bottom=.025,
        height_ratios=[.25, 1, 1, 1, 1],
        width_ratios=[float(config['layout']['context_width_ratio']), 1, .075, 1, 1, 1, .095],
        hspace=float(config['layout']['hspace']), wspace=float(config['layout']['wspace']))

    title_ax = fig.add_subplot(block[0, :]); title_ax.axis('off')
    split_name = {'train': 'training', 'val': 'validation', 'test': 'test'}[case['split']]
    title_ax.text(0, .5, f"Selected {split_name} sample · {_short_fire_name(case['fire_name'])}",
                  ha='left', va='center', fontsize=config['fonts']['title_size'], fontweight='semibold')

    context_ax = fig.add_subplot(block[1, 0])
    _draw_context(context_ax, context, config)
    note_ax = fig.add_subplot(block[2, 0]); note_ax.axis('off')
    note_ax.text(.03, .96, f"Observed $t$={case['latest_input_timestamp']}\nFuture $t$={case['target_timestamp']}\n"
        f"GT active {case['target_active_fraction']:.1%}\nWind: lowest {len(config['_wind_levels'])} levels\n"
        f"Rows 2–4: |pred − GT|\nOrange box = patch",
        transform=note_ax.transAxes, va='top', ha='left', fontsize=config['fonts']['tick_size'], color='#444444')
    fuel = np.asarray(context['surface_fuel_full'], dtype=np.float32)
    patch = context['surface_fuel_patch_bounds']
    y0, x0, patch_h, patch_w = (int(patch[key]) for key in ('y0', 'x0', 'height', 'width'))
    if (fuel.ndim != 2 or not np.isfinite(fuel).all() or fuel.shape[0] < y0+patch_h
            or fuel.shape[1] < x0+patch_w or min(y0, x0) < 0
            or (patch_h, patch_w) != target[0].shape):
        raise ValueError(f'Full-fire surface fuel or evaluated patch is invalid for {case["sample_id"]}')
    positive_fuel = fuel[fuel > 0]
    fuel_vmax = max(float(np.percentile(positive_fuel, 99)), 1e-6) if positive_fuel.size else 1.0
    fuel_ax = fig.add_subplot(block[3:5, 0])
    _image_axis(fuel_ax, fuel, cmap=config['surface_fuel']['cmap'], vmin=0, vmax=fuel_vmax)
    from matplotlib.patches import Rectangle
    fuel_ax.add_patch(Rectangle((x0-.5, y0-.5), patch_w, patch_h, fill=False,
                                edgecolor='white', linewidth=2.2, zorder=6))
    fuel_ax.add_patch(Rectangle((x0-.5, y0-.5), patch_w, patch_h, fill=False,
                                edgecolor='#D55E00', linewidth=1.2, zorder=7))
    fuel_ax.set_title('Full surface fuel', fontsize=config['fonts']['label_size'], pad=3)
    fuel_position = fuel_ax.get_position()
    fuel_ax.set_position([fuel_position.x0 - float(config['surface_fuel']['left_shift']),
                          fuel_position.y0, fuel_position.width, fuel_position.height])

    gt_mask = target[2] > FIRE_MASK_THRESHOLD
    row_specs = (('mask', 2, 'Active fire'), ('surface', 0, 'Surface fuel'),
                 ('canopy', 1, 'Canopy fuel'), ('energy_log', 3, 'HRR'))
    for row_index, (quantity, channel, row_label) in enumerate(row_specs, start=1):
        gt_vmin, gt_vmax = limits[quantity]
        gt_cmap = config['mask']['cmap'] if quantity == 'mask' else config['continuous']['cmaps'][quantity]
        gt_ax = fig.add_subplot(block[row_index, 1])
        gt_im = _image_axis(gt_ax, target[channel], cmap=gt_cmap, vmin=gt_vmin, vmax=gt_vmax)
        gt_ax.set_ylabel(row_label, labelpad=3)
        if row_index == 1:
            gt_ax.set_title(display_names[0], pad=2)

        if quantity == 'mask':
            model_images = {key: 1/(1+np.exp(-np.clip(predictions[key][2], -60, 60))) for key in model_keys}
            model_cmap, model_vmax = gt_cmap, 1.0
        else:
            model_images = absolute_error_maps(target, predictions, model_keys, channel)
            model_cmap = config['absolute_error']['cmap']
            model_vmax = absolute_error_limit(model_images, float(config['absolute_error']['color_percentile']))
            gt_cax = fig.add_subplot(block[row_index, 2])
            gt_bar = fig.colorbar(gt_im, cax=gt_cax)
            gt_bar.ax.tick_params(labelsize=config['fonts']['colorbar_size'], length=2, pad=1)
            if quantity == 'energy_log':
                gt_bar.ax.set_ylabel('log(1+MW)', fontsize=config['fonts']['colorbar_size'], labelpad=2)

        for column, key in enumerate(model_keys, start=3):
            values = model_images[key]
            ax = fig.add_subplot(block[row_index, column])
            model_im = _image_axis(ax, values, cmap=model_cmap, vmin=0, vmax=model_vmax)
            if row_index == 1:
                ax.set_title(display_names[column-2], pad=2)
            if quantity == 'mask' and config['mask']['show_gt_perimeter'] and gt_mask.any() and not gt_mask.all():
                ax.contour(gt_mask.astype(float), levels=[.5], colors='white',
                           linewidths=float(config['mask']['perimeter_linewidth']))
            if quantity == 'mask' and config['mask']['show_threshold_contour']:
                if np.any(values < .5) and np.any(values > .5):
                    ax.contour(values, levels=[.5], colors='#44e1d5', linewidths=.5)
        model_cax = fig.add_subplot(block[row_index, 6])
        exceeds = quantity != 'mask' and any(np.any(values > model_vmax) for values in model_images.values())
        model_bar = fig.colorbar(model_im, cax=model_cax, extend='max' if exceeds else 'neither')
        model_bar.ax.tick_params(labelsize=config['fonts']['colorbar_size'], length=2, pad=1)
        if quantity != 'mask':
            unit = '|Δ log(1+MW)|' if quantity == 'energy_log' else '|Pred − GT|'
            model_bar.ax.set_ylabel(unit, fontsize=config['fonts']['colorbar_size'], labelpad=2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(config['figure']['dpi']))
    plt.close(fig)
