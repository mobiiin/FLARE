#!/usr/bin/env python3
"""Generate publication-quality, full-frame CAWFE appendix figures.

This script intentionally operates on ``fires/<fire>/frames/frame_*.npz`` and
``targets/h10/<fire>/target_*.npz``.  It never imports or calls the patch
sampler: every panel is the complete processed CAWFE spatial domain.

Examples
--------
python scripts/plot_cawfe_appendix_channels.py \
    --figure channel_examples --selection-seed 42 --num-candidates 10

python scripts/plot_cawfe_appendix_channels.py \
    --figure dynamic_channels --selection-seed 42 --num-candidates 10
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from src.config import load_config
from src.data.dataset import resolve_engineered_feature_slices
from src.data.processed_dataset import load_channel_manifest, load_dataset_manifest


DEFAULT_DATASET_ROOT = Path("/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")
DEFAULT_OUTPUT_DIR = Path("figures/cawfe_appendix_candidates")
FIGURE_TYPES = ("channel_examples", "dynamic_channels")

# The atmospheric ordering is the established repository convention in
# scripts/visualize_patch_cache.py::_default_channel_name.  The offsets are
# validated against the configured 10-variable-per-level layout below.
ATMOSPHERIC_VARIABLES = ("U", "V", "W", "T", "QV", "QC", "QR", "QI", "QS", "P")
U_OFFSET = ATMOSPHERIC_VARIABLES.index("U")
V_OFFSET = ATMOSPHERIC_VARIABLES.index("V")
W_OFFSET = ATMOSPHERIC_VARIABLES.index("W")
THETA_OFFSET = ATMOSPHERIC_VARIABLES.index("T")

WIND_UNITS = r"$\mathrm{m\,s^{-1}}$"
TEMPERATURE_UNITS = r"$\mathrm{K}$"
FLUX_UNITS = r"$\mathrm{W\,m^{-2}}$"
# The processed manifests do not attach a dimensional unit to fuel.  Keep the
# unnormalised physical values and state that they are in native CAWFE units
# rather than inventing a unit that is absent from the authoritative metadata.
FUEL_UNITS = "native CAWFE fuel units"


@dataclass(frozen=True)
class ChannelLayout:
	"""Canonical raw/engineered channel positions resolved from config."""

	variables_per_level: int
	num_vertical_levels: int
	low_level_indices: tuple[int, ...]
	surface_sensible_flux: int
	surface_fuel: int
	canopy_fuel: int
	horizontal_speed_start: int
	low_level_mean_speed: int
	step_surface_consumed: int
	cumulative_surface_consumed: int


@dataclass(frozen=True)
class Candidate:
	"""One complete fire-domain state and its canonical horizon target."""

	fire_name: str
	split: str
	current_index: int
	future_index: int
	frame_path: Path
	target_path: Path
	shape: tuple[int, int]
	score: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
	if not isinstance(value, Mapping):
		raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
	return value


def _resolve_layout(config: Mapping[str, Any], channel_manifest: Mapping[str, Any]) -> ChannelLayout:
	"""Resolve and aggressively validate every plotted channel."""
	layout = _mapping(config.get("channel_layout"), "channel_layout")
	atmospheric = _mapping(config.get("atmospheric_features"), "atmospheric_features")
	energy = _mapping(config.get("energy_release"), "energy_release")

	variables_per_level = int(atmospheric["variables_per_level"])
	num_vertical_levels = int(atmospheric["num_vertical_levels"])
	low_levels = tuple(int(value) for value in atmospheric["low_level_indices"])
	atmospheric_span = tuple(int(value) for value in layout["atmospheric_channels"])
	if variables_per_level != len(ATMOSPHERIC_VARIABLES):
		raise ValueError(
			"The established CAWFE variable order contains 10 variables per level, "
			f"but the resolved config declares {variables_per_level}."
		)
	if atmospheric_span != (0, num_vertical_levels * variables_per_level - 1):
		raise ValueError(
			"Configured atmospheric span is incompatible with the level-major CAWFE layout: "
			f"span={atmospheric_span}, levels={num_vertical_levels}, width={variables_per_level}."
		)
	if not low_levels or min(low_levels) < 0 or max(low_levels) >= num_vertical_levels:
		raise ValueError(f"Invalid low_level_indices: {low_levels}")

	flux_channels = tuple(int(value) for value in layout["flux_channels"])
	fuel_channels = tuple(int(value) for value in layout["fuel_channels"])
	if flux_channels != (80, 81, 82, 83):
		raise ValueError(f"Unexpected canonical flux channel layout: {flux_channels}")
	if fuel_channels != (84, 85):
		raise ValueError(f"Unexpected canonical fuel channel layout: {fuel_channels}")
	surface_flux = int(energy["surface_sensible_flux_channel"])
	if surface_flux != flux_channels[0]:
		raise ValueError("Surface sensible heat-flux definitions disagree between config sections.")
	surface_fuel = int(layout["surface_fuel_channel"])
	canopy_fuel = int(layout["canopy_fuel_channel"])
	if (surface_fuel, canopy_fuel) != fuel_channels:
		raise ValueError("Surface/canopy fuel definitions disagree with channel_layout.fuel_channels.")

	num_raw = int(channel_manifest["num_raw_channels"])
	if num_raw != 86:
		raise ValueError(f"Expected 86 canonical raw CAWFE channels, found {num_raw}.")
	slices = resolve_engineered_feature_slices(config, num_raw)
	required = ("horizontal_wind_speed", "low_level_mean_wind_speed", "step_consumed_fuel", "cumulative_consumed_fuel")
	missing = [name for name in required if name not in slices]
	if missing:
		raise KeyError(f"Required canonical engineered channels are missing: {missing}")
	horizontal_slice = slices["horizontal_wind_speed"]
	low_mean_slice = slices["low_level_mean_wind_speed"]
	step_slice = slices["step_consumed_fuel"]
	cumulative_slice = slices["cumulative_consumed_fuel"]
	if any(value.stop is None or value.start is None for value in (horizontal_slice, low_mean_slice, step_slice, cumulative_slice)):
		raise ValueError("Canonical engineered-channel slices must have finite bounds.")
	if horizontal_slice.stop - horizontal_slice.start != num_vertical_levels:
		raise ValueError("Horizontal-wind-speed engineered slice does not match the number of levels.")
	if low_mean_slice.stop - low_mean_slice.start != 1:
		raise ValueError("Expected exactly one canonical low-level mean wind-speed channel.")
	if step_slice.stop - step_slice.start != len(fuel_channels):
		raise ValueError("Step-wise consumed-fuel slice does not match surface/canopy fuel channels.")
	if cumulative_slice.stop - cumulative_slice.start != len(fuel_channels):
		raise ValueError("Cumulative consumed-fuel slice does not match surface/canopy fuel channels.")

	return ChannelLayout(
		variables_per_level=variables_per_level,
		num_vertical_levels=num_vertical_levels,
		low_level_indices=low_levels,
		surface_sensible_flux=surface_flux,
		surface_fuel=surface_fuel,
		canopy_fuel=canopy_fuel,
		horizontal_speed_start=int(horizontal_slice.start),
		low_level_mean_speed=int(low_mean_slice.start),
		step_surface_consumed=int(step_slice.start),
		cumulative_surface_consumed=int(cumulative_slice.start),
	)


def _target_path(root: Path, horizon: int, fire: str, current_index: int) -> Path:
	return (
		root
		/ "targets"
		/ f"h{horizon}"
		/ fire
		/ f"target_current_{current_index:06d}_future_{current_index + horizon:06d}.npz"
	)


def _frame_path(root: Path, fire: str, current_index: int) -> Path:
	return root / "fires" / fire / "frames" / f"frame_{current_index:06d}.npz"


def _load_target(path: Path) -> dict[str, np.ndarray]:
	with np.load(path, allow_pickle=False) as archive:
		required = ("surface_consumed", "canopy_consumed", "fire_mask", "energy_log")
		missing = [key for key in required if key not in archive]
		if missing:
			raise KeyError(f"Canonical target archive {path} is missing {missing}")
		return {key: np.asarray(archive[key]) for key in required}


def _load_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
	with np.load(path, allow_pickle=False) as archive:
		if "x_raw" not in archive or "x_engineered" not in archive:
			raise KeyError(f"Full-frame archive must contain x_raw and x_engineered: {path}")
		raw = np.asarray(archive["x_raw"], dtype=np.float32)
		engineered = np.asarray(archive["x_engineered"], dtype=np.float32)
	return raw, engineered


def _finite_fraction(*arrays: np.ndarray) -> float:
	total = sum(int(np.asarray(array).size) for array in arrays)
	finite = sum(int(np.isfinite(np.asarray(array)).sum()) for array in arrays)
	return float(finite / max(total, 1))


def _positive_percentile(array: np.ndarray, percentile: float = 99.0) -> float:
	values = np.asarray(array, dtype=np.float64)
	positive = values[np.isfinite(values) & (values > 0)]
	return float(np.percentile(positive, percentile)) if positive.size else 0.0


def _target_quality(target: Mapping[str, np.ndarray], *, require_activity: bool) -> tuple[bool, float]:
	surface = np.asarray(target["surface_consumed"], dtype=np.float32)
	canopy = np.asarray(target["canopy_consumed"], dtype=np.float32)
	mask = np.asarray(target["fire_mask"], dtype=bool)
	hrr = np.asarray(target["energy_log"], dtype=np.float32)
	if len({surface.shape, canopy.shape, mask.shape, hrr.shape}) != 1 or surface.ndim != 2:
		return False, -math.inf
	if _finite_fraction(surface, canopy, hrr) < 0.9999:
		return False, -math.inf
	active_pixels = int(mask.sum())
	surface_pixels = int(np.count_nonzero(surface > 0))
	hrr_pixels = int(np.count_nonzero(hrr > 0))
	if require_activity and (active_pixels == 0 or surface_pixels == 0 or hrr_pixels == 0):
		return False, -math.inf
	if active_pixels == 0 and surface_pixels == 0 and hrr_pixels == 0:
		return False, -math.inf
	# Data-only score used only to reject empty/degenerate states and find clear
	# examples within a fire.  No predictions or model outputs are inspected.
	score = (
		math.log1p(active_pixels)
		+ math.log1p(surface_pixels)
		+ math.log1p(hrr_pixels)
		+ 0.15 * math.log1p(_positive_percentile(surface))
		+ 0.10 * math.log1p(_positive_percentile(hrr))
	)
	return True, float(score)


def _frame_fields(raw: np.ndarray, engineered: np.ndarray, channels: ChannelLayout) -> dict[str, np.ndarray]:
	if raw.ndim != 3 or engineered.ndim != 3:
		raise ValueError(f"Full frames must be C,H,W; got raw={raw.shape}, engineered={engineered.shape}")
	if raw.shape[1:] != engineered.shape[1:]:
		raise ValueError(f"Raw/engineered spatial shapes disagree: {raw.shape} vs {engineered.shape}")
	width = channels.variables_per_level
	level0 = 0
	u0 = raw[level0 * width + U_OFFSET]
	v0 = raw[level0 * width + V_OFFSET]
	w0 = raw[level0 * width + W_OFFSET]
	theta0 = raw[level0 * width + THETA_OFFSET]
	low_u = np.mean(np.stack([raw[level * width + U_OFFSET] for level in channels.low_level_indices]), axis=0)
	low_v = np.mean(np.stack([raw[level * width + V_OFFSET] for level in channels.low_level_indices]), axis=0)
	low_speed_derived = np.sqrt(low_u * low_u + low_v * low_v + 1.0e-6).astype(np.float32)
	low_speed_canonical = engineered[channels.low_level_mean_speed]
	if not np.allclose(low_speed_derived, low_speed_canonical, rtol=2.0e-5, atol=2.0e-5):
		raise ValueError("Canonical low-level mean wind-speed channel disagrees with configured U/V levels.")
	return {
		"u0": u0,
		"v0": v0,
		"speed0": np.sqrt(u0 * u0 + v0 * v0 + 1.0e-6).astype(np.float32),
		"w0": w0,
		"theta0": theta0,
		"low_u": low_u.astype(np.float32),
		"low_v": low_v.astype(np.float32),
		"low_speed": np.asarray(low_speed_canonical, dtype=np.float32),
		"surface_flux": raw[channels.surface_sensible_flux],
		"surface_fuel": raw[channels.surface_fuel],
		"step_surface": engineered[channels.step_surface_consumed],
		"cumulative_surface": engineered[channels.cumulative_surface_consumed],
	}


def _frame_quality(fields: Mapping[str, np.ndarray]) -> tuple[bool, float]:
	relevant = [np.asarray(fields[name]) for name in ("speed0", "w0", "theta0", "surface_flux", "surface_fuel", "step_surface", "cumulative_surface")]
	if _finite_fraction(*relevant) < 0.9999:
		return False, -math.inf
	variations = {
		"wind": float(np.nanstd(fields["speed0"])),
		"vertical": float(np.nanstd(fields["w0"])),
		"temperature": float(np.nanstd(fields["theta0"])),
		"flux": float(np.nanstd(fields["surface_flux"])),
		"fuel": float(np.nanstd(fields["surface_fuel"])),
	}
	if variations["wind"] <= 1.0e-6 or variations["flux"] <= 1.0e-6 or variations["fuel"] <= 1.0e-6:
		return False, -math.inf
	if np.count_nonzero(np.asarray(fields["step_surface"]) > 0) == 0:
		return False, -math.inf
	if np.count_nonzero(np.asarray(fields["cumulative_surface"]) > 0) == 0:
		return False, -math.inf
	score = sum(math.log1p(max(value, 0.0)) for value in variations.values())
	score += 0.25 * math.log1p(np.count_nonzero(np.asarray(fields["step_surface"]) > 0))
	return True, float(score)


def _screen_indices(frame_count: int, horizon: int, rng: np.random.Generator, count: int = 24) -> list[int]:
	valid_count = int(frame_count) - int(horizon)
	if valid_count <= 0:
		return []
	count = min(int(count), valid_count)
	# Random selection is deterministic under selection_seed; two interior
	# anchors reduce the chance that a short simulation contributes only an edge.
	indices = set(int(value) for value in rng.choice(valid_count, size=count, replace=False))
	indices.add(valid_count // 3)
	indices.add((2 * valid_count) // 3)
	return sorted(indices)


def _best_candidate_for_fire(
	root: Path,
	fire_name: str,
	fire_info: Mapping[str, Any],
	horizon: int,
	patch_size: int,
	rng: np.random.Generator,
	channels: ChannelLayout,
	figure_type: str,
	excluded_indices: set[int] | None = None,
) -> Candidate | None:
	excluded = excluded_indices or set()
	shortlist: list[tuple[float, int, Path, dict[str, np.ndarray]]] = []
	for current_index in _screen_indices(int(fire_info["num_processed_frames"]), horizon, rng):
		if current_index in excluded:
			continue
		target_path = _target_path(root, horizon, fire_name, current_index)
		if not target_path.is_file():
			continue
		try:
			target = _load_target(target_path)
		except (OSError, KeyError, ValueError):
			continue
		valid, target_score = _target_quality(target, require_activity=figure_type == "dynamic_channels")
		if valid:
			shortlist.append((target_score, current_index, target_path, target))
	if not shortlist:
		return None

	# Only load the three strongest full-frame input archives for each fire.
	# Target files are much smaller, so this keeps deterministic screening fast.
	for target_score, current_index, target_path, target in sorted(shortlist, reverse=True)[:3]:
		frame_path = _frame_path(root, fire_name, current_index)
		if not frame_path.is_file():
			continue
		try:
			raw, engineered = _load_frame(frame_path)
			fields = _frame_fields(raw, engineered, channels)
		except (OSError, KeyError, ValueError):
			continue
		height, width = (int(value) for value in raw.shape[1:])
		# This is the critical pre-patchification assertion.
		if height <= patch_size or width <= patch_size:
			continue
		if engineered.shape[1:] != (height, width):
			continue
		if any(np.asarray(value).shape != (height, width) for value in target.values()):
			continue
		valid, frame_score = _frame_quality(fields)
		if not valid:
			continue
		return Candidate(
			fire_name=fire_name,
			split=str(fire_info.get("split", "unknown")),
			current_index=current_index,
			future_index=current_index + horizon,
			frame_path=frame_path,
			target_path=target_path,
			shape=(height, width),
			score=float(target_score + frame_score),
		)
	return None


def _select_candidates(
	root: Path,
	dataset_manifest: Mapping[str, Any],
	channels: ChannelLayout,
	figure_type: str,
	selection_seed: int,
	num_candidates: int,
	horizon: int,
	patch_size: int,
) -> list[Candidate]:
	fires_payload = _mapping(dataset_manifest.get("fires"), "dataset_manifest.fires")
	fires = sorted(str(name) for name in fires_payload)
	if not fires:
		raise ValueError("No fires are present in the processed dataset manifest.")
	rng = np.random.default_rng(int(selection_seed))
	order = list(np.asarray(fires, dtype=object)[rng.permutation(len(fires))])
	selected: list[Candidate] = []
	used: dict[str, set[int]] = {}

	# First pass: at most one full-domain state per fire, maximizing fire diversity.
	for fire_name_obj in order:
		fire_name = str(fire_name_obj)
		info = _mapping(fires_payload[fire_name], f"dataset_manifest.fires[{fire_name!r}]")
		candidate = _best_candidate_for_fire(
			root,
			fire_name,
			info,
			horizon,
			patch_size,
			rng,
			channels,
			figure_type,
		)
		if candidate is not None:
			selected.append(candidate)
			used.setdefault(fire_name, set()).add(candidate.current_index)
		if len(selected) >= num_candidates:
			return selected

	# Extra rounds support requests larger than the number of eligible fires.
	for _round in range(1, 6):
		made_progress = False
		for fire_name_obj in order:
			fire_name = str(fire_name_obj)
			info = _mapping(fires_payload[fire_name], f"dataset_manifest.fires[{fire_name!r}]")
			candidate = _best_candidate_for_fire(
				root,
				fire_name,
				info,
				horizon,
				patch_size,
				rng,
				channels,
				figure_type,
				excluded_indices=used.get(fire_name, set()),
			)
			if candidate is not None:
				selected.append(candidate)
				used.setdefault(fire_name, set()).add(candidate.current_index)
				made_progress = True
			if len(selected) >= num_candidates:
				return selected
		if not made_progress:
			break
	raise RuntimeError(
		f"Only {len(selected)} nondegenerate full-frame {figure_type} candidates were found; "
		f"requested {num_candidates}."
	)


def _finite_values(array: np.ndarray) -> np.ndarray:
	values = np.asarray(array, dtype=np.float64)
	return values[np.isfinite(values)]


def _sequential_limits(array: np.ndarray, *, nonnegative_sparse: bool = False) -> tuple[float, float]:
	values = _finite_values(array)
	if values.size == 0:
		raise ValueError("Cannot plot a field containing no finite values.")
	if nonnegative_sparse:
		positive = values[values > 0]
		if positive.size:
			vmax = float(np.percentile(positive, 99.0))
			if not np.isfinite(vmax) or vmax <= 0:
				vmax = float(positive.max())
			return 0.0, max(vmax, np.finfo(np.float32).eps)
		return 0.0, 1.0
	vmin, vmax = (float(value) for value in np.percentile(values, [1.0, 99.0]))
	if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
		center = float(values[0]) if values.size else 0.0
		delta = max(abs(center) * 0.01, 1.0e-6)
		return center - delta, center + delta
	return vmin, vmax


def _signed_limit(array: np.ndarray) -> float:
	values = np.abs(_finite_values(array))
	if values.size == 0:
		raise ValueError("Cannot plot a field containing no finite values.")
	vmax = float(np.percentile(values, 99.0))
	return max(vmax, np.finfo(np.float32).eps)


def _configure_style() -> None:
	plt.rcParams.update(
		{
			"font.family": "serif",
			"font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
			"mathtext.fontset": "stix",
			"axes.titleweight": "regular",
			"axes.linewidth": 0.45,
			"figure.facecolor": "white",
			"axes.facecolor": "white",
			"savefig.facecolor": "white",
		}
	)


def _clean_fire_name(name: str) -> str:
	cleaned = re.sub(r"__keepz_\d+$", "", name, flags=re.IGNORECASE)
	cleaned = cleaned.replace("__", " ").replace("_", " ")
	return " ".join(cleaned.split())


def _safe_fire_name(name: str) -> str:
	return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _draw_image(
	fig: plt.Figure,
	ax: plt.Axes,
	array: np.ndarray,
	title: str,
	units: str,
	*,
	cmap: str,
	vmin: float | None = None,
	vmax: float | None = None,
	norm: Any | None = None,
) -> Any:
	image = ax.imshow(
		array,
		origin="lower",
		interpolation="nearest",
		aspect="equal",
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		norm=norm,
		rasterized=True,
	)
	ax.set_title(title, fontsize=6.4, pad=2.2)
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_linewidth(0.4)
	colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.052, pad=0.045, aspect=22)
	colorbar.ax.tick_params(labelsize=4.5, length=1.6, width=0.35, pad=1.0)
	colorbar.outline.set_linewidth(0.35)
	colorbar.set_label(units, fontsize=5.0, labelpad=0.7)
	return image


def _draw_quiver(ax: plt.Axes, u: np.ndarray, v: np.ndarray) -> None:
	height, width = u.shape
	stride = int(np.clip(round(max(height, width) / 16.0), 8, 16))
	y = np.arange(0, height, stride)
	x = np.arange(0, width, stride)
	ax.quiver(
		x,
		y,
		u[np.ix_(y, x)],
		v[np.ix_(y, x)],
		color="white",
		edgecolor="black",
		linewidth=0.18,
		pivot="middle",
		angles="xy",
		minshaft=1.2,
		minlength=0.6,
		width=0.004,
		zorder=3,
	)


def _plot_channel_examples(
	candidate: Candidate,
	fields: Mapping[str, np.ndarray],
	output_path: Path,
) -> None:
	fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.55), constrained_layout=False)
	fig.subplots_adjust(left=0.035, right=0.985, bottom=0.075, top=0.915, wspace=0.18, hspace=0.33)

	vmin, vmax = _sequential_limits(fields["speed0"])
	_draw_image(fig, axes[0, 0], fields["speed0"], "(a) Low-level horizontal wind", WIND_UNITS, cmap="viridis", vmin=vmin, vmax=vmax)
	_draw_quiver(axes[0, 0], fields["u0"], fields["v0"])

	w_limit = _signed_limit(fields["w0"])
	_draw_image(
		fig,
		axes[0, 1],
		fields["w0"],
		"(b) Vertical velocity",
		WIND_UNITS,
		cmap="RdBu_r",
		norm=TwoSlopeNorm(vmin=-w_limit, vcenter=0.0, vmax=w_limit),
	)

	vmin, vmax = _sequential_limits(fields["theta0"])
	_draw_image(fig, axes[0, 2], fields["theta0"], "(c) Potential temperature", TEMPERATURE_UNITS, cmap="cividis", vmin=vmin, vmax=vmax)

	flux_values = _finite_values(fields["surface_flux"])
	if flux_values.min(initial=0.0) < 0 < flux_values.max(initial=0.0):
		flux_limit = _signed_limit(fields["surface_flux"])
		_draw_image(
			fig,
			axes[0, 3],
			fields["surface_flux"],
			"(d) Surface sensible heat flux",
			FLUX_UNITS,
			cmap="RdBu_r",
			norm=TwoSlopeNorm(vmin=-flux_limit, vcenter=0.0, vmax=flux_limit),
		)
	else:
		vmin, vmax = _sequential_limits(fields["surface_flux"], nonnegative_sparse=bool(np.all(flux_values >= 0)))
		_draw_image(fig, axes[0, 3], fields["surface_flux"], "(d) Surface sensible heat flux", FLUX_UNITS, cmap="inferno", vmin=vmin, vmax=vmax)

	vmin, vmax = _sequential_limits(fields["surface_fuel"])
	_draw_image(fig, axes[1, 0], fields["surface_fuel"], "(e) Surface fuel", FUEL_UNITS, cmap="YlGn", vmin=vmin, vmax=vmax)

	vmin, vmax = _sequential_limits(fields["step_surface"], nonnegative_sparse=True)
	_draw_image(fig, axes[1, 1], fields["step_surface"], "(f) Step-wise surface consumption", FUEL_UNITS, cmap="magma", vmin=vmin, vmax=vmax)

	vmin, vmax = _sequential_limits(fields["low_speed"])
	_draw_image(fig, axes[1, 2], fields["low_speed"], "(g) Wind speed and direction", WIND_UNITS, cmap="viridis", vmin=vmin, vmax=vmax)
	_draw_quiver(axes[1, 2], fields["low_u"], fields["low_v"])

	vmin, vmax = _sequential_limits(fields["cumulative_surface"], nonnegative_sparse=True)
	_draw_image(fig, axes[1, 3], fields["cumulative_surface"], "(h) Cumulative surface consumption", FUEL_UNITS, cmap="magma", vmin=vmin, vmax=vmax)

	fig.suptitle(
		f"{_clean_fire_name(candidate.fire_name)} — full CAWFE domain at $t={candidate.current_index}$",
		fontsize=7.8,
		y=0.975,
	)
	fig.savefig(output_path, dpi=600, format="png", facecolor="white")
	plt.close(fig)


def _plot_dynamic_channels(
	candidate: Candidate,
	fields: Mapping[str, np.ndarray],
	target: Mapping[str, np.ndarray],
	output_path: Path,
	horizon: int,
) -> None:
	fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.55), constrained_layout=False)
	fig.subplots_adjust(left=0.067, right=0.985, bottom=0.075, top=0.915, wspace=0.18, hspace=0.33)

	vmin, vmax = _sequential_limits(fields["low_speed"])
	_draw_image(fig, axes[0, 0], fields["low_speed"], "(a) Low-level wind speed", WIND_UNITS, cmap="viridis", vmin=vmin, vmax=vmax)
	w_limit = _signed_limit(fields["w0"])
	_draw_image(
		fig,
		axes[0, 1],
		fields["w0"],
		"(b) Vertical velocity",
		WIND_UNITS,
		cmap="RdBu_r",
		norm=TwoSlopeNorm(vmin=-w_limit, vcenter=0.0, vmax=w_limit),
	)
	flux_values = _finite_values(fields["surface_flux"])
	if flux_values.min(initial=0.0) < 0 < flux_values.max(initial=0.0):
		flux_limit = _signed_limit(fields["surface_flux"])
		_draw_image(
			fig,
			axes[0, 2],
			fields["surface_flux"],
			"(c) Surface sensible heat flux",
			FLUX_UNITS,
			cmap="RdBu_r",
			norm=TwoSlopeNorm(vmin=-flux_limit, vcenter=0.0, vmax=flux_limit),
		)
	else:
		vmin, vmax = _sequential_limits(fields["surface_flux"], nonnegative_sparse=bool(np.all(flux_values >= 0)))
		_draw_image(fig, axes[0, 2], fields["surface_flux"], "(c) Surface sensible heat flux", FLUX_UNITS, cmap="inferno", vmin=vmin, vmax=vmax)
	vmin, vmax = _sequential_limits(fields["surface_fuel"])
	_draw_image(fig, axes[0, 3], fields["surface_fuel"], "(d) Surface fuel", FUEL_UNITS, cmap="YlGn", vmin=vmin, vmax=vmax)

	_draw_image(fig, axes[1, 0], np.asarray(target["fire_mask"], dtype=np.float32), "(e) Active fire", "binary target (0/1)", cmap="Greys", vmin=0.0, vmax=1.0)
	vmin, vmax = _sequential_limits(target["surface_consumed"], nonnegative_sparse=True)
	_draw_image(fig, axes[1, 1], target["surface_consumed"], "(f) Surface fuel consumed", FUEL_UNITS, cmap="magma", vmin=vmin, vmax=vmax)
	vmin, vmax = _sequential_limits(target["canopy_consumed"], nonnegative_sparse=True)
	_draw_image(fig, axes[1, 2], target["canopy_consumed"], "(g) Canopy fuel consumed", FUEL_UNITS, cmap="magma", vmin=vmin, vmax=vmax)
	vmin, vmax = _sequential_limits(target["energy_log"], nonnegative_sparse=True)
	_draw_image(
		fig,
		axes[1, 3],
		target["energy_log"],
		r"(h) $\log(1 + P_{\mathrm{HRR}})$",
		r"$\log(1 + P_{\mathrm{HRR}}/\mathrm{MW})$",
		cmap="inferno",
		vmin=vmin,
		vmax=vmax,
	)

	fig.text(0.015, 0.695, r"Input state at $t$", rotation=90, va="center", ha="center", fontsize=6.0)
	fig.text(
		0.015,
		0.285,
		f"Forecast labels at $t+{horizon}$ min / over interval",
		rotation=90,
		va="center",
		ha="center",
		fontsize=5.5,
	)
	fig.suptitle(
		f"{_clean_fire_name(candidate.fire_name)} — full CAWFE domain, $t={candidate.current_index}$ to $t+{horizon}$ min",
		fontsize=7.8,
		y=0.975,
	)
	fig.savefig(output_path, dpi=600, format="png", facecolor="white")
	plt.close(fig)


def _verify_candidate(
	candidate: Candidate,
	raw: np.ndarray,
	engineered: np.ndarray,
	target: Mapping[str, np.ndarray],
	patch_size: int,
) -> None:
	height, width = candidate.shape
	if raw.shape[1:] != (height, width) or engineered.shape[1:] != (height, width):
		raise AssertionError("Loaded full-frame shape changed after candidate selection.")
	if height <= patch_size or width <= patch_size:
		raise AssertionError(
			f"Full-frame assertion failed: H={height}, W={width}, configured patch_size={patch_size}."
		)
	for name, array in target.items():
		if np.asarray(array).shape != (height, width):
			raise AssertionError(f"Target {name} shape {np.asarray(array).shape} does not match {(height, width)}")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Generate full-domain, pre-patchification CAWFE appendix channel figures (PNG only)."
	)
	parser.add_argument("--figure", required=True, choices=FIGURE_TYPES)
	parser.add_argument("--selection-seed", type=int, default=42)
	parser.add_argument("--num-candidates", type=int, default=10)
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--dataset-root", help="Override processed full-frame dataset root.")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	return parser


def run(args: argparse.Namespace) -> list[Path]:
	if int(args.num_candidates) <= 0:
		raise ValueError("--num-candidates must be positive.")
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	processed = _mapping(config.get("processed_dataset"), "processed_dataset")
	root = Path(args.dataset_root or processed.get("root", DEFAULT_DATASET_ROOT)).expanduser().resolve()
	output_dir = Path(args.output_dir).expanduser().resolve()
	if output_dir.suffix:
		raise ValueError("--output-dir must be a directory; generated outputs are PNG files only.")
	output_dir.mkdir(parents=True, exist_ok=True)

	dataset_manifest = load_dataset_manifest(root)
	channel_manifest = load_channel_manifest(root)
	channels = _resolve_layout(config, channel_manifest)
	target_config = _mapping(config.get("target_construction"), "target_construction")
	horizon = int(target_config["horizon"])
	patch_config = _mapping(processed.get("patch_index"), "processed_dataset.patch_index")
	patch_size = int(patch_config["patch_size"])
	if horizon != 10:
		raise ValueError(f"The requested appendix semantics require a 10-minute horizon; resolved horizon={horizon}.")

	_configure_style()
	candidates = _select_candidates(
		root,
		dataset_manifest,
		channels,
		str(args.figure),
		int(args.selection_seed),
		int(args.num_candidates),
		horizon,
		patch_size,
	)
	outputs: list[Path] = []
	for candidate_index, candidate in enumerate(candidates):
		raw, engineered = _load_frame(candidate.frame_path)
		target = _load_target(candidate.target_path)
		_verify_candidate(candidate, raw, engineered, target, patch_size)
		fields = _frame_fields(raw, engineered, channels)
		filename = (
			f"cawfe_{args.figure}_{candidate_index:02d}_{_safe_fire_name(candidate.fire_name)}"
			f"_t{candidate.current_index:06d}.png"
		)
		output_path = output_dir / filename
		if args.figure == "channel_examples":
			_plot_channel_examples(candidate, fields, output_path)
		else:
			_plot_dynamic_channels(candidate, fields, target, output_path, horizon)
		outputs.append(output_path)
		print("\nCANDIDATE")
		print(f"  figure type: {args.figure}")
		print(f"  candidate index: {candidate_index:02d}")
		print(f"  fire name: {candidate.fire_name}")
		print(f"  split: {candidate.split}")
		print(f"  timestep: {candidate.current_index}")
		print(f"  full frame shape H x W: {candidate.shape[0]} x {candidate.shape[1]}")
		print(f"  source file/frame: {candidate.frame_path}")
		print("  selected atmospheric level: z0 for U/V/W/potential temperature; canonical mean z[0,1,2] where labeled low-level mean")
		if args.figure == "dynamic_channels":
			print(f"  target full-frame shape: {np.asarray(target['fire_mask']).shape}")
			print(f"  target source: {candidate.target_path}")
		print("  PATCHIFICATION USED: NO")
		print("  FULL FRAME VISUALIZED: YES")
		print(f"  saved: {output_path}")

	print("\nCAWFE APPENDIX VISUALIZATION")
	print("\nExisting visualization code reused:")
	print("  YES")
	print("  path: scripts/visualize_engineered_frames.py; scripts/visualize_targets.py")
	print("\nFull pre-patch frames used:")
	print("  YES")
	print("\nTraining patch sampler used:")
	print("  NO")
	figure_1_count = len(list(output_dir.glob("cawfe_channel_examples_*.png")))
	figure_2_count = len(list(output_dir.glob("cawfe_dynamic_channels_*.png")))
	print("\nFigure 1 candidates:")
	print(f"  {figure_1_count}")
	print("\nFigure 2 candidates:")
	print(f"  {figure_2_count}")
	print("\nOutput directory:")
	print(f"  {output_dir}")
	print("\nOutput format:")
	print("  PNG ONLY")
	print("\nCommands to generate both candidate sets:")
	print("  python scripts/plot_cawfe_appendix_channels.py \\")
	print("      --figure channel_examples \\")
	print("      --selection-seed 42 \\")
	print("      --num-candidates 10")
	print("  python scripts/plot_cawfe_appendix_channels.py \\")
	print("      --figure dynamic_channels \\")
	print("      --selection-seed 42 \\")
	print("      --num-candidates 10")
	return outputs


def main() -> None:
	run(_build_parser().parse_args())


if __name__ == "__main__":
	main()
