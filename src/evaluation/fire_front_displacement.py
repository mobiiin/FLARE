"""Like-for-like interval burn-footprint reconstruction and figure helpers."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from scripts.export_qualitative_forecasts import raw_case
from src.data.processed_targets import build_processed_target
from src.evaluation.fire_activity import FIRE_MASK_THRESHOLD, PREDICTED_FIRE_THRESHOLD

CASE_TITLES = {
    'directional': 'Directional spread',
    'curved': 'Curved / multi-front spread',
    'difficult': 'Challenging physical regime',
}


@lru_cache(maxsize=16)
def _raw_frame(path: str) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'Cannot reconstruct the preceding burn interval: missing raw frame {source}')
    with np.load(source, allow_pickle=False) as archive:
        if 'x_raw' not in archive:
            raise KeyError(f'Cannot reconstruct the preceding burn interval: x_raw missing in {source}')
        return np.asarray(archive['x_raw'], dtype=np.float32)


@lru_cache(maxsize=16)
def _area(root: str, fire: str) -> np.ndarray:
    fire_root = Path(root) / 'fires' / fire
    manifest = json.loads((fire_root / 'fire_manifest.json').read_text(encoding='utf-8'))
    path = fire_root / manifest['geometry']['area_2d_path']
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)


def _patch(array: np.ndarray, patch: dict) -> np.ndarray:
    y0, x0, height, width = (int(patch[key]) for key in ('y0', 'x0', 'height', 'width'))
    result = np.asarray(array[y0:y0+height, x0:x0+width])
    if result.shape != (height, width):
        raise ValueError(f'Patch {patch} is outside an array with shape {array.shape}')
    return result


def _frame_path(root: Path, fire: str, index: int) -> Path:
    return root / 'fires' / fire / 'frames' / f'frame_{index:06d}.npz'


def reconstruct_current_mask(root: Path, record: dict, target_config: dict, thresholds: dict) -> np.ndarray:
    """Build [t-h,t] mask using the exact target builder and frozen thresholds."""
    t, horizon = int(record['current_index']), int(record['horizon'])
    if t - horizon < 0:
        raise ValueError(f'Cannot reconstruct [t-{horizon},t] for {record["sample_id"]}: preceding index is negative')
    fire = record['fire_name']
    previous = _raw_frame(str(_frame_path(root, fire, t-horizon)))
    current = _raw_frame(str(_frame_path(root, fire, t)))
    target = build_processed_target(previous, current, _area(str(root), fire), target_config, thresholds)
    return _patch(target['fire_mask'], record['patch']).astype(bool)


def validate_future_target(root: Path, record: dict, target_config: dict, thresholds: dict) -> np.ndarray:
    """Require reconstructed [t,t+h] target to equal the stored canonical target."""
    t, horizon = int(record['current_index']), int(record['horizon'])
    fire = record['fire_name']
    current = _raw_frame(str(_frame_path(root, fire, t)))
    future = _raw_frame(str(_frame_path(root, fire, t+horizon)))
    rebuilt = build_processed_target(current, future, _area(str(root), fire), target_config, thresholds)
    _, stored, _ = raw_case(root, record, include_input=False, include_terrain=False)
    for name, channel in (('surface_consumed', 0), ('canopy_consumed', 1), ('fire_mask', 2), ('energy_log', 3)):
        reconstructed = _patch(rebuilt[name], record['patch'])
        exact = np.array_equal(reconstructed, stored[channel])
        if name == 'energy_log':
            # Float32 energy aggregation can vary by a few ULPs across NumPy builds.
            exact = np.allclose(reconstructed, stored[channel], rtol=0, atol=1e-6)
        if not exact:
            raise RuntimeError(f'Reconstructed {name} differs from the canonical future target: {record["sample_id"]}')
    return stored[2] > FIRE_MASK_THRESHOLD


def physical_context(root: Path, record: dict, config: dict) -> dict:
    """Read only current-frame wind and the matching static terrain patch."""
    fire, t, patch = record['fire_name'], int(record['current_index']), record['patch']
    frame = _raw_frame(str(_frame_path(root, fire, t)))
    atmospheric = config.get('atmospheric_features', {})
    if 'low_level_indices' not in atmospheric or 'variables_per_level' not in atmospheric:
        raise ValueError('Resolved dataset config must explicitly define low-level wind channels.')
    levels = [int(value) for value in atmospheric['low_level_indices']]
    stride = int(atmospheric['variables_per_level'])
    if not levels or min(levels) < 0 or max(levels)*stride+1 >= frame.shape[0]:
        raise ValueError('Configured low-level U/V indices do not fit the raw frame.')
    u = np.mean(np.stack([_patch(frame[level*stride], patch) for level in levels]), axis=0).astype(np.float32)
    v = np.mean(np.stack([_patch(frame[level*stride+1], patch) for level in levels]), axis=0).astype(np.float32)
    terrain_path = root / 'fires' / fire / 'terrain/terrain_features.npy'
    if not terrain_path.is_file():
        raise FileNotFoundError(f'Relative-elevation terrain is missing: {terrain_path}')
    terrain = np.load(terrain_path, allow_pickle=False)
    elevation = _patch(terrain[0], patch).astype(np.float32)
    return {'wind_u': u, 'wind_v': v, 'terrain_elevation': elevation,
            'wind_source': str(_frame_path(root, fire, t)), 'terrain_source': str(terrain_path)}


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    y, x = np.nonzero(mask)
    return (float(x.mean()), float(y.mean())) if x.size else None


def _border_margin(mask: np.ndarray) -> int:
    y, x = np.nonzero(mask)
    return int(min(x.min(), y.min(), mask.shape[1]-1-x.max(), mask.shape[0]-1-y.max())) if x.size else -1


def mask_descriptors(current: np.ndarray, future: np.ndarray, context: dict) -> dict:
    current, future = np.asarray(current, dtype=bool), np.asarray(future, dtype=bool)
    area = int(future.sum())
    components, component_count = label(future, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components[future], minlength=component_count+1)[1:]
    perimeter = int(boundary_pixels(future).sum())
    first, second = _centroid(current), _centroid(future)
    displacement = float(np.hypot(second[0]-first[0], second[1]-first[1])) if first and second else None
    terrain = context['terrain_elevation']
    gy, gx = np.gradient(terrain)
    speed = np.hypot(context['wind_u'], context['wind_v'])
    valid = speed > 1e-6
    direction_variation = (float(1 - np.hypot(np.mean(context['wind_u'][valid]/speed[valid]),
                                              np.mean(context['wind_v'][valid]/speed[valid])))
                           if valid.any() else 0.0)
    return {
        'current_active_pixels': int(current.sum()),
        'current_border_margin_pixels': _border_margin(current),
        'future_border_margin_pixels': _border_margin(future),
        'future_active_pixels': area,
        'current_active_fraction': float(current.mean()),
        'future_active_fraction': float(future.mean()),
        'centroid_current_xy': list(first) if first else None,
        'centroid_future_xy': list(second) if second else None,
        'centroid_displacement_pixels': displacement,
        'centroid_displacement_patch_widths': displacement / future.shape[1] if displacement is not None else None,
        'component_count': int(component_count),
        'dominant_component_fraction': float(sizes.max()/area) if area else 0.0,
        'perimeter_pixels': perimeter,
        'boundary_complexity': float(perimeter*perimeter/(4*np.pi*area)) if area else None,
        'terrain_elevation_std': float(np.std(terrain)),
        'terrain_slope_std': float(np.std(np.hypot(gx, gy))),
        'wind_direction_variation': direction_variation,
    }


def boundary_metrics(gt: np.ndarray, predicted: np.ndarray) -> dict:
    gt_edge, pred_edge = boundary_pixels(gt), boundary_pixels(predicted)
    if not gt_edge.any() or not pred_edge.any():
        return {'symmetric_mean_boundary_distance_pixels': None, 'hausdorff_distance_pixels': None,
                'centroid_error_pixels': None}
    pred_dist = distance_transform_edt(~pred_edge)
    gt_dist = distance_transform_edt(~gt_edge)
    a, b = pred_dist[gt_edge], gt_dist[pred_edge]
    gt_centroid, pred_centroid = _centroid(gt), _centroid(predicted)
    return {
        'symmetric_mean_boundary_distance_pixels': float((a.mean()+b.mean())/2),
        'hausdorff_distance_pixels': float(max(a.max(), b.max())),
        'centroid_error_pixels': float(np.hypot(gt_centroid[0]-pred_centroid[0], gt_centroid[1]-pred_centroid[1])),
    }


def _style(config: dict):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': config['fonts']['family'], 'font.size': config['fonts']['base_size'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'figure.facecolor': config['figure']['background'],
                         'savefig.facecolor': config['figure']['background']})
    return plt


def _readable_fire(name: str) -> str:
    return name.split('__')[0].replace('_', ' ').replace('CHIMNEYTOPS2', 'Chimney Tops 2').title()


def _base_axis(ax, current: np.ndarray, future: np.ndarray, context: dict, config: dict, *, prediction=None,
               model_key=None, show_arrow=False):
    height, width = current.shape
    ax.set_facecolor(config['figure']['panel_background'])
    ax.set_xlim(-.5, width-.5)
    ax.set_ylim(height-.5, -.5)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    elevation = context['terrain_elevation']
    terrain = config['terrain']
    if terrain['enabled'] and float(np.ptp(elevation)) > 1e-9:
        vmin, vmax = np.percentile(elevation, [2, 98])
        if vmax > vmin:
            ax.imshow(elevation, cmap=terrain['cmap'], vmin=vmin, vmax=vmax,
                      alpha=float(terrain['alpha']), interpolation='bilinear', origin='upper')
        levels = np.linspace(float(elevation.min()), float(elevation.max()), int(terrain['levels'])+2)[1:-1]
        ax.contour(elevation, levels=levels, colors='#63737a',
                   linewidths=float(terrain['contour_linewidth']), alpha=float(terrain['contour_alpha']))
    if config['wind']['enabled']:
        wind = config['wind']; stride = int(wind['stride'])
        if stride < 1:
            raise ValueError('wind.stride must be positive')
        yy, xx = np.mgrid[stride//2:height:stride, stride//2:width:stride]
        u = context['wind_u'][yy, xx].copy()
        v = context['wind_v'][yy, xx].copy()
        if wind.get('normalize_direction', False):
            speed = np.hypot(u, v)
            u = np.divide(u, speed, out=np.zeros_like(u), where=speed > 1e-6)
            v = np.divide(v, speed, out=np.zeros_like(v), where=speed > 1e-6)
        ax.quiver(xx, yy, u, v, color=wind['color'], alpha=float(wind['alpha']),
                  angles='xy', scale_units='xy', scale=float(wind['scale']),
                  width=float(wind['width']), pivot='middle', zorder=3)
    # Keep the dashed observed boundary visible when all three masks nearly coincide.
    for mask, key in ((future, 'gt'), (prediction,
                         f'prediction_{model_key}' if model_key else None), (current, 'current')):
        if mask is None:
            continue
        style = config['lines'][key]
        if np.any(mask) and not np.all(mask):
            ax.contour(np.asarray(mask, dtype=float), levels=[.5], colors=[style['color']],
                       linestyles=[style['linestyle']], linewidths=float(style['linewidth']), zorder=4)
    if show_arrow:
        first, second = _centroid(current), _centroid(future)
        if first is not None and second is not None:
            ax.annotate('', xy=second, xytext=first,
                        arrowprops={'arrowstyle':'->', 'color':'#777777', 'lw':0.75, 'alpha':0.65}, zorder=5)
    for spine in ax.spines.values():
        spine.set_linewidth(.5); spine.set_color('#b5b5b5')


def draw_preview(cases: list[dict], data: list[dict], output: Path, config: dict) -> None:
    plt = _style(config)
    fig, axes = plt.subplots(3, 2, figsize=(config['figure']['width'], 8.8), squeeze=False)
    fig.subplots_adjust(left=.10, right=.97, top=.93, bottom=.06, wspace=.20, hspace=.32)
    for col, title in enumerate(('Preceding interval', 'Future ground truth')):
        axes[0, col].set_title(title, fontsize=config['fonts']['title_size'], pad=14)
    for row, (case, item) in enumerate(zip(cases, data)):
        current, future, context = item['current_mask'], item['future_mask'], item['context']
        for col in (0, 1):
            ax = axes[row, col]
            _base_axis(ax, current if col == 0 else np.zeros_like(current),
                       future if col == 1 else np.zeros_like(future), context, config)
            if col == 0:
                ax.text(-.03, 1.03, f"({chr(97+row)}) {CASE_TITLES[case['case_type']]} · {_readable_fire(case['fire_name'])}",
                        transform=ax.transAxes, fontsize=config['fonts']['row_size'], fontweight='semibold',
                        ha='left', va='bottom', clip_on=False)
    fig.suptitle('GT/context-only burn-footprint candidates · held-out test', fontsize=config['fonts']['title_size'])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def draw_final(cases: list[dict], data: list[dict], output_dir: Path, config: dict, *,
               show_centroid_arrow=False, annotate_metrics=False, metrics=None,
               formats=('pdf', 'svg', 'png'), output_name='fire_front_displacement') -> None:
    from matplotlib.lines import Line2D
    plt = _style(config)
    fig, axes = plt.subplots(3, 2, figsize=(config['figure']['width'], config['figure']['height']), squeeze=False)
    fig.subplots_adjust(left=.09, right=.97, top=.88, bottom=.12, wspace=.08, hspace=.30)
    for col, title in enumerate(('FLARE baseline', 'FLARE final')):
        axes[0, col].set_title(title, fontsize=config['fonts']['title_size'], pad=27, fontweight='semibold')
    for row, (case, item) in enumerate(zip(cases, data)):
        current, future, context = item['current_mask'], item['future_mask'], item['context']
        for col, key in enumerate(('baseline', 'final')):
            ax = axes[row, col]
            _base_axis(ax, current, future, context, config, prediction=item['predicted_masks'][key],
                       model_key=key, show_arrow=show_centroid_arrow)
            if col == 0:
                ax.text(-.03, 1.035, f"({chr(97+row)}) {CASE_TITLES[case['case_type']]} · {_readable_fire(case['fire_name'])}",
                        transform=ax.transAxes, fontsize=config['fonts']['row_size'], fontweight='semibold',
                        ha='left', va='bottom', clip_on=False)
            if annotate_metrics and metrics is not None:
                found = next(m for m in metrics if m['sample_id'] == case['sample_id'] and m['model_key'] == key)
                ax.text(.97, .97, f"Dice {found['dice']:.3f} · IoU {found['iou']:.3f}", transform=ax.transAxes,
                        ha='right', va='top', fontsize=config['fonts']['note_size'],
                        bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':2})
    styles = config['lines']
    handles = [Line2D([0],[0], color=styles[key]['color'], linestyle=styles[key]['linestyle'],
                      linewidth=styles[key]['linewidth'], label=label)
               for key,label in (('current','Observed interval boundary'),('gt','Ground-truth future boundary'),
                                 ('prediction_baseline','Predicted future boundary'))]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.53,.045), fontsize=config['fonts']['legend_size'])
    fig.suptitle('Burn-footprint boundary displacement · held-out test',
                 fontsize=config['fonts']['title_size'], y=.985)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        if extension not in ('pdf', 'svg', 'png'):
            raise ValueError(f'Unsupported figure format: {extension}')
        fig.savefig(output_dir / f'{output_name}.{extension}', dpi=int(config['figure']['dpi']))
    plt.close(fig)
