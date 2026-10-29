"""Exact streaming full-validation evaluation for trained forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from src.data.dataset import metadata_batch_to_list
from src.evaluation.fire_activity import (
    ACTIVE_FRACTION_BIN_NAMES,
    ACTIVE_FRACTION_BOUNDARIES,
    ACTIVE_FRACTION_THRESHOLD,
    FIRE_MASK_CHANNEL,
    FIRE_MASK_THRESHOLD,
    PREDICTED_FIRE_THRESHOLD,
    active_fraction_bin_index,
)
from src.evaluation.no_fire_metrics import FullValidationNoFireAccumulator, classify_target_masks
from src.evaluation.validation_subset import (
    DEFAULT_QUALITATIVE_INDEX_PATH,
    QUALITATIVE_SELECTION_SEED,
    ensure_qualitative_validation_samples,
)
from src.training.batch_utils import unpack_batch
from src.training.hardware import autocast_context
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader
from src.training.model_outputs import extract_prediction


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _sqrt_mean(sum_squared: float, count: int) -> float | None:
    return math.sqrt(max(0.0, float(sum_squared) / float(count))) if count > 0 else None


@dataclass
class FullValidationAccumulator:
    """Accumulate exact global segmentation/regression metrics in float64."""

    active_fraction_threshold: float = ACTIVE_FRACTION_THRESHOLD
    total_patches: int = 0
    fire_patches: int = 0
    no_fire_patches: int = 0
    total_pixels: int = 0
    active_pixels: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    surface_absolute_error: float = 0.0
    surface_squared_error: float = 0.0
    canopy_absolute_error: float = 0.0
    canopy_squared_error: float = 0.0
    energy_absolute_error: float = 0.0
    energy_squared_error: float = 0.0
    energy_mw_absolute_error: float = 0.0
    energy_mw_squared_error: float = 0.0
    active_surface_absolute_error: float = 0.0
    active_surface_squared_error: float = 0.0
    active_canopy_absolute_error: float = 0.0
    active_canopy_squared_error: float = 0.0
    active_energy_absolute_error: float = 0.0
    active_energy_squared_error: float = 0.0
    no_fire: FullValidationNoFireAccumulator = field(default_factory=FullValidationNoFireAccumulator)

    def __post_init__(self) -> None:
        self.no_fire.active_fraction_threshold = float(self.active_fraction_threshold)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] < 4:
            raise ValueError(
                f"Full validation expects matching (B, >=4, H, W) tensors, got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        prediction = prediction.detach().to(dtype=torch.float64)
        target = target.detach().to(dtype=torch.float64)
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Full-validation prediction/target contains NaN or Inf.")

        classification = classify_target_masks(
            target,
            fire_threshold=FIRE_MASK_THRESHOLD,
            active_fraction_threshold=self.active_fraction_threshold,
        )
        fire = classification["has_fire"]
        batch_size = int(target.shape[0])
        batch_fire = int(fire.sum().item())
        self.total_patches += batch_size
        self.fire_patches += batch_fire
        self.no_fire_patches += batch_size - batch_fire

        true_mask = target[:, FIRE_MASK_CHANNEL] > FIRE_MASK_THRESHOLD
        probability = torch.sigmoid(prediction[:, FIRE_MASK_CHANNEL])
        predicted_mask = probability > PREDICTED_FIRE_THRESHOLD
        self.total_pixels += int(true_mask.numel())
        self.active_pixels += int(true_mask.sum().item())
        self.true_positive += int((predicted_mask & true_mask).sum().item())
        self.false_positive += int((predicted_mask & ~true_mask).sum().item())
        self.false_negative += int((~predicted_mask & true_mask).sum().item())

        for channel, prefix in ((0, "surface"), (1, "canopy"), (3, "energy")):
            error = prediction[:, channel] - target[:, channel]
            absolute = error.abs()
            setattr(self, f"{prefix}_absolute_error", getattr(self, f"{prefix}_absolute_error") + float(absolute.sum().item()))
            setattr(self, f"{prefix}_squared_error", getattr(self, f"{prefix}_squared_error") + float(error.square().sum().item()))
            if true_mask.any():
                setattr(
                    self,
                    f"active_{prefix}_absolute_error",
                    getattr(self, f"active_{prefix}_absolute_error") + float(absolute[true_mask].sum().item()),
                )
                setattr(
                    self,
                    f"active_{prefix}_squared_error",
                    getattr(self, f"active_{prefix}_squared_error") + float(error.square()[true_mask].sum().item()),
                )
        predicted_energy_mw = torch.clamp(torch.expm1(prediction[:, 3]), min=0.0)
        target_energy_mw = torch.clamp(torch.expm1(target[:, 3]), min=0.0)
        energy_mw_error = predicted_energy_mw - target_energy_mw
        if not torch.isfinite(energy_mw_error).all():
            raise ValueError("Full-validation MW energy conversion contains NaN or Inf.")
        self.energy_mw_absolute_error += float(energy_mw_error.abs().sum().item())
        self.energy_mw_squared_error += float(energy_mw_error.square().sum().item())
        self.no_fire.update(prediction.to(dtype=torch.float32), target.to(dtype=torch.float32))

    def finalize(self) -> dict[str, Any]:
        if self.total_patches != self.fire_patches + self.no_fire_patches:
            raise RuntimeError("Full-validation patch counts do not sum to the evaluated total.")
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        union = self.true_positive + self.false_positive + self.false_negative
        metrics: dict[str, Any] = {
            "full_val_total_patch_count": self.total_patches,
            "full_val_fire_patch_count": self.fire_patches,
            "full_val_no_fire_patch_count": self.no_fire_patches,
            "full_val_fire_patch_percent": 100.0 * self.fire_patches / self.total_patches if self.total_patches else 0.0,
            "full_val_no_fire_patch_percent": 100.0 * self.no_fire_patches / self.total_patches if self.total_patches else 0.0,
            "full_val_no_fire_percent": 100.0 * self.no_fire_patches / self.total_patches if self.total_patches else 0.0,
            "full_val_dice": _safe_divide(2 * self.true_positive, denominator),
            "full_val_iou": _safe_divide(self.true_positive, union),
            "full_val_precision": _safe_divide(self.true_positive, self.true_positive + self.false_positive),
            "full_val_recall": _safe_divide(self.true_positive, self.true_positive + self.false_negative),
            "full_val_surface_mae": _safe_divide(self.surface_absolute_error, self.total_pixels),
            "full_val_surface_rmse": _sqrt_mean(self.surface_squared_error, self.total_pixels),
            "full_val_canopy_mae": _safe_divide(self.canopy_absolute_error, self.total_pixels),
            "full_val_canopy_rmse": _sqrt_mean(self.canopy_squared_error, self.total_pixels),
            "full_val_energy_log_mae": _safe_divide(self.energy_absolute_error, self.total_pixels),
            "full_val_energy_log_rmse": _sqrt_mean(self.energy_squared_error, self.total_pixels),
            "full_val_energy_mw_mae": _safe_divide(self.energy_mw_absolute_error, self.total_pixels),
            "full_val_energy_mw_rmse": _sqrt_mean(self.energy_mw_squared_error, self.total_pixels),
            "full_val_active_surface_mae": _safe_divide(self.active_surface_absolute_error, self.active_pixels),
            "full_val_active_surface_rmse": _sqrt_mean(self.active_surface_squared_error, self.active_pixels),
            "full_val_active_canopy_mae": _safe_divide(self.active_canopy_absolute_error, self.active_pixels),
            "full_val_active_canopy_rmse": _sqrt_mean(self.active_canopy_squared_error, self.active_pixels),
            "full_val_active_energy_log_mae": _safe_divide(self.active_energy_absolute_error, self.active_pixels),
            "full_val_active_energy_log_rmse": _sqrt_mean(self.active_energy_squared_error, self.active_pixels),
            "full_val_mask_true_positive_pixel_count": self.true_positive,
            "full_val_mask_false_positive_pixel_count": self.false_positive,
            "full_val_mask_false_negative_pixel_count": self.false_negative,
            "full_val_active_pixel_count": self.active_pixels,
            "full_val_total_pixel_count": self.total_pixels,
        }
        metrics.update(self.no_fire.finalize())
        # Compatibility aliases match training metric names while keeping the
        # explicit full_val_ scope required by ablation reporting.
        metrics.update(
            {
                "full_val_mask_dice": metrics["full_val_dice"],
                "full_val_mask_iou": metrics["full_val_iou"],
                "full_val_mask_precision": metrics["full_val_precision"],
                "full_val_mask_recall": metrics["full_val_recall"],
                "full_val_surface_consumed_mae": metrics["full_val_surface_mae"],
                "full_val_surface_consumed_rmse": metrics["full_val_surface_rmse"],
                "full_val_canopy_consumed_mae": metrics["full_val_canopy_mae"],
                "full_val_canopy_consumed_rmse": metrics["full_val_canopy_rmse"],
                "full_val_active_surface_consumed_mae": metrics["full_val_active_surface_mae"],
                "full_val_active_surface_consumed_rmse": metrics["full_val_active_surface_rmse"],
                "full_val_active_canopy_consumed_mae": metrics["full_val_active_canopy_mae"],
                "full_val_active_canopy_consumed_rmse": metrics["full_val_active_canopy_rmse"],
            }
        )
        return metrics


def _per_fire_metrics(accumulator: FullValidationAccumulator) -> dict[str, Any]:
    metrics = accumulator.finalize()
    return {
        "sample_count": metrics["full_val_total_patch_count"],
        "fire_patch_count": metrics["full_val_fire_patch_count"],
        "no_fire_patch_count": metrics["full_val_no_fire_patch_count"],
        "dice": metrics["full_val_dice"],
        "iou": metrics["full_val_iou"],
        "precision": metrics["full_val_precision"],
        "recall": metrics["full_val_recall"],
        "surface_mae": metrics["full_val_surface_mae"],
        "surface_rmse": metrics["full_val_surface_rmse"],
        "canopy_mae": metrics["full_val_canopy_mae"],
        "canopy_rmse": metrics["full_val_canopy_rmse"],
        "energy_log_mae": metrics["full_val_energy_log_mae"],
        "energy_log_rmse": metrics["full_val_energy_log_rmse"],
        "energy_mw_mae": metrics["full_val_energy_mw_mae"],
        "energy_mw_rmse": metrics["full_val_energy_mw_rmse"],
        "active_canopy_mae": metrics["full_val_active_canopy_mae"],
        "active_energy_log_mae": metrics["full_val_active_energy_log_mae"],
        "no_fire_mask_probability": metrics["full_val_no_fire_mask_prob_mean"],
        "no_fire_mask_false_positive_rate": metrics["full_val_no_fire_mask_false_positive_rate"],
        "no_fire_patch_false_positive_rate": metrics["full_val_no_fire_patch_false_positive_rate"],
        "no_fire_mask_false_positive_patch_count": metrics["full_val_no_fire_false_positive_patch_count"],
        "no_fire_surface_abs_pred_mean": metrics["full_val_no_fire_surface_abs_pred_mean"],
        "no_fire_canopy_abs_pred_mean": metrics["full_val_no_fire_canopy_abs_pred_mean"],
        "no_fire_energy_log_abs_pred_mean": metrics["full_val_no_fire_energy_log_abs_pred_mean"],
        "no_fire_energy_log_pred_mean": metrics["full_val_no_fire_energy_log_pred_mean"],
    }


def _activity_bin_metrics(accumulator: FullValidationAccumulator) -> dict[str, Any]:
    metrics = accumulator.finalize()
    return {
        "patch_count": metrics["full_val_total_patch_count"],
        "fire_patch_count": metrics["full_val_fire_patch_count"],
        "no_fire_patch_count": metrics["full_val_no_fire_patch_count"],
        "dice": metrics["full_val_dice"],
        "iou": metrics["full_val_iou"],
        "surface_mae": metrics["full_val_surface_mae"],
        "canopy_mae": metrics["full_val_canopy_mae"],
        "energy_log_mae": metrics["full_val_energy_log_mae"],
        "active_canopy_mae": metrics["full_val_active_canopy_mae"],
    }


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    """Write a complete CSV atomically, preserving a stable union schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    if not columns:
        raise ValueError(f"Cannot write a CSV without fields: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sample_metric_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metadata_items: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute one machine-readable metric row for every validation patch."""

    prediction = prediction.detach().to(dtype=torch.float64)
    target = target.detach().to(dtype=torch.float64)
    if len(metadata_items) != int(target.shape[0]):
        raise RuntimeError("Per-sample metrics require one metadata item per prediction.")
    true_mask = target[:, FIRE_MASK_CHANNEL] > FIRE_MASK_THRESHOLD
    probability = torch.sigmoid(prediction[:, FIRE_MASK_CHANNEL])
    predicted_mask = probability > PREDICTED_FIRE_THRESHOLD
    active_pixels = true_mask.flatten(1).sum(dim=1)
    pixels_per_patch = int(true_mask[0].numel())
    active_fraction = active_pixels.to(torch.float64) / float(pixels_per_patch)
    true_positive = (predicted_mask & true_mask).flatten(1).sum(dim=1).to(torch.float64)
    false_positive = (predicted_mask & ~true_mask).flatten(1).sum(dim=1).to(torch.float64)
    false_negative = (~predicted_mask & true_mask).flatten(1).sum(dim=1).to(torch.float64)

    def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> list[float | None]:
        return [float(n / d) if float(d) > 0.0 else None for n, d in zip(numerator.tolist(), denominator.tolist())]

    dice = safe_ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative)
    iou = safe_ratio(true_positive, true_positive + false_positive + false_negative)
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    regression: dict[str, torch.Tensor] = {}
    for channel, name in ((0, "surface"), (1, "canopy"), (3, "energy_log")):
        error = prediction[:, channel] - target[:, channel]
        regression[f"{name}_mae"] = error.abs().flatten(1).mean(dim=1)
        regression[f"{name}_rmse"] = error.square().flatten(1).mean(dim=1).sqrt()
    energy_mw_error = torch.clamp(torch.expm1(prediction[:, 3]), min=0.0) - torch.clamp(torch.expm1(target[:, 3]), min=0.0)
    regression["energy_mw_mae"] = energy_mw_error.abs().flatten(1).mean(dim=1)
    regression["energy_mw_rmse"] = energy_mw_error.square().flatten(1).mean(dim=1).sqrt()

    rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(metadata_items):
        active_count = int(active_pixels[index].item())
        fraction = float(active_fraction[index].item())
        activity_bin = ACTIVE_FRACTION_BIN_NAMES[active_fraction_bin_index(fraction)]
        active_canopy_error = (prediction[index, 1] - target[index, 1]).abs()[true_mask[index]]
        active_energy_error = (prediction[index, 3] - target[index, 3]).abs()[true_mask[index]]
        no_fire = active_count == 0
        row: dict[str, Any] = {
            "model_name": model_name,
            "seed": int(seed),
            "sample_id": str(metadata.get("sample_id", index)),
            "fire_name": str(metadata.get("fire_name", metadata.get("fire", "unknown"))),
            "fire_activity_bin": activity_bin,
            "activity_bin": activity_bin,
            "target_active_fraction": fraction,
            "active_fraction": fraction,
            "active_pixel_count": active_count,
            "pixel_count": pixels_per_patch,
            "is_no_fire": int(no_fire),
            "mask_dice": dice[index],
            "mask_iou": iou[index],
            "dice": dice[index],
            "iou": iou[index],
            "precision": precision[index],
            "recall": recall[index],
            "predicted_fire_fraction": float(predicted_mask[index].to(torch.float64).mean().item()),
            "mean_mask_probability": float(probability[index].mean().item()),
            "mean_surface_prediction": float(prediction[index, 0].mean().item()),
            "mean_canopy_prediction": float(prediction[index, 1].mean().item()),
            "mean_energy_log_prediction": float(prediction[index, 3].mean().item()),
            "mask_false_positive_pixel_count": int(false_positive[index].item()),
            "active_canopy_mae": float(active_canopy_error.mean().item()) if active_count else None,
            "active_energy_log_mae": float(active_energy_error.mean().item()) if active_count else None,
            "no_fire_mask_probability": float(probability[index].mean().item()) if no_fire else None,
            "no_fire_mask_false_positive_rate": float(predicted_mask[index].to(torch.float64).mean().item()) if no_fire else None,
            "no_fire_patch_false_positive": int(predicted_mask[index].any().item()) if no_fire else None,
            "no_fire_surface_abs_pred_mean": float(prediction[index, 0].abs().mean().item()) if no_fire else None,
            "no_fire_canopy_abs_pred_mean": float(prediction[index, 1].abs().mean().item()) if no_fire else None,
            "no_fire_energy_log_abs_pred_mean": float(prediction[index, 3].abs().mean().item()) if no_fire else None,
        }
        for key, values in regression.items():
            row[key] = float(values[index].item())
        rows.append(row)
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _model_call(model: torch.nn.Module, inputs: torch.Tensor, terrain: torch.Tensor | None):
    return model(inputs) if terrain is None else model(inputs, terrain=terrain)


def evaluate_full_validation(
    *,
    model: torch.nn.Module,
    val_loader: Any,
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: Any,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    checkpoint_epoch: int | None,
    expected_counts: Mapping[str, Any] | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Evaluate every validation sample once and persist exact run artifacts."""

    if bool(getattr(val_loader, "drop_last", False)):
        raise RuntimeError("Full validation requires drop_last=false.")
    if not isinstance(getattr(val_loader, "sampler", None), SequentialSampler):
        raise RuntimeError(f"Full validation requires deterministic no-shuffle sampling, got {type(getattr(val_loader, 'sampler', None)).__name__}.")
    expected_samples = len(val_loader.dataset)
    if expected_samples <= 0:
        raise RuntimeError("Full validation dataset is empty.")

    training_config = config.get("training", {}) if isinstance(config.get("training"), Mapping) else {}
    validation_config = training_config.get("validation", {}) if isinstance(training_config.get("validation"), Mapping) else {}
    full_config = validation_config.get("full", {}) if isinstance(validation_config.get("full"), Mapping) else {}
    final_config = config.get("final_training", {}) if isinstance(config.get("final_training"), Mapping) else {}
    model_name = str(final_config.get("finalist", config.get("run", {}).get("architecture", "model")))
    seed = int(training_config.get("seed", config.get("seed", 42)))
    save_analysis_data = bool(full_config.get("save_analysis_data", bool(final_config)))

    qualitative_payload: Mapping[str, Any] | None = None
    qualitative_by_id: dict[str, Mapping[str, Any]] = {}
    if save_analysis_data:
        qualitative_payload = ensure_qualitative_validation_samples(
            val_loader.dataset,
            config,
            output_path=full_config.get("qualitative_index_path", DEFAULT_QUALITATIVE_INDEX_PATH),
            per_group=int(full_config.get("qualitative_samples_per_group", 4)),
            seed=int(full_config.get("qualitative_seed", QUALITATIVE_SELECTION_SEED)),
            logger=logger,
        )
        qualitative_by_id = {
            str(item["sample_id"]): item for item in qualitative_payload.get("selected_samples", [])
        }

    model.eval()
    normalizer = build_input_normalizer_for_loader(val_loader, device, int(config.get("model", {}).get("input_channels", 0)))
    overall = FullValidationAccumulator()
    per_fire: dict[str, FullValidationAccumulator] = {}
    per_activity = {name: FullValidationAccumulator() for name in ACTIVE_FRACTION_BIN_NAMES}
    sample_ids: set[str] = set()
    sample_id_count = 0
    sample_rows: list[dict[str, Any]] = []
    qualitative_arrays: dict[str, dict[str, Any]] = {}
    warmup_batches = min(5, max(0, len(val_loader) - 1))
    timed_seconds = 0.0
    timed_samples = 0
    timed_batches = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(val_loader, start=1):
            if device.type == "cuda" and batch_index > warmup_batches:
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter() if batch_index > warmup_batches else None
            x_raw, y_raw, extra = unpack_batch(batch)
            terrain_raw = extra.get("terrain")
            x = apply_input_normalization(x_raw.to(device, non_blocking=True), normalizer)
            y = y_raw.to(device, non_blocking=True).float()
            terrain = terrain_raw.to(device, non_blocking=True) if terrain_raw is not None else None
            with autocast_context(device, amp_dtype):
                output = _model_call(model, x, terrain)
            prediction = extract_prediction(output).float()
            if inference_start is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timed_seconds += time.perf_counter() - inference_start
                timed_samples += int(y.shape[0])
                timed_batches += 1
            overall.update(prediction, y)

            activity = classify_target_masks(y)
            grouped_activity: dict[str, list[int]] = {}
            for sample_index, active_fraction in enumerate(activity["active_fraction"].detach().cpu().tolist()):
                bin_name = ACTIVE_FRACTION_BIN_NAMES[active_fraction_bin_index(float(active_fraction))]
                grouped_activity.setdefault(bin_name, []).append(sample_index)
            for bin_name, indices in grouped_activity.items():
                per_activity[bin_name].update(prediction[indices], y[indices])

            metadata_batch = extra.get("metadata")
            metadata_items = metadata_batch_to_list(metadata_batch, batch_size=int(y.shape[0])) if isinstance(metadata_batch, Mapping) else []
            if metadata_items and len(metadata_items) != int(y.shape[0]):
                raise RuntimeError("Validation metadata batch length does not match prediction batch length.")
            if save_analysis_data and len(metadata_items) != int(y.shape[0]):
                raise RuntimeError("Paper-analysis exports require sample metadata for every validation patch.")

            grouped: dict[str, list[int]] = {}
            for sample_index, metadata in enumerate(metadata_items):
                fire_name = str(metadata.get("fire_name", metadata.get("fire", "unknown")))
                grouped.setdefault(fire_name, []).append(sample_index)
                sample_id = metadata.get("sample_id")
                if sample_id is not None:
                    sample_id_count += 1
                    text_id = str(sample_id)
                    if text_id in sample_ids:
                        raise RuntimeError(f"Full validation visited sample_id more than once: {text_id}")
                    sample_ids.add(text_id)
                    if text_id in qualitative_by_id:
                        target_item = y[sample_index].detach().cpu().numpy().astype(np.float32, copy=False)
                        prediction_item = prediction[sample_index].detach().cpu()
                        qualitative_arrays[text_id] = {
                            "sample_id": text_id,
                            "fire_name": fire_name,
                            "activity_bin": str(qualitative_by_id[text_id]["activity_bin"]),
                            "target_surface": target_item[0],
                            "target_canopy": target_item[1],
                            "target_mask": target_item[FIRE_MASK_CHANNEL],
                            "target_energy_log": target_item[3],
                            "pred_surface": prediction_item[0].numpy().astype(np.float32, copy=False),
                            "pred_canopy": prediction_item[1].numpy().astype(np.float32, copy=False),
                            "pred_mask_probability": torch.sigmoid(prediction_item[FIRE_MASK_CHANNEL]).numpy().astype(np.float32, copy=False),
                            "pred_energy_log": prediction_item[3].numpy().astype(np.float32, copy=False),
                        }
            for fire_name, indices in grouped.items():
                if fire_name not in per_fire:
                    per_fire[fire_name] = FullValidationAccumulator()
                per_fire[fire_name].update(prediction[indices], y[indices])
            if save_analysis_data:
                new_rows = _sample_metric_rows(prediction, y, metadata_items, model_name=model_name, seed=seed)
                if config.get("model", {}).get("terrain_integration") is not None:
                    if terrain_raw is None or terrain_raw.shape[1] != 4:
                        raise RuntimeError("Terrain ablation requires four terrain channels in full validation.")
                    for row, patch_terrain in zip(new_rows, terrain_raw):
                        # Dataset channels: relative elevation, slope magnitude, slope_x, slope_y.
                        elevation = patch_terrain[0].float()
                        slope = patch_terrain[1].float()
                        row.update(mean_elevation=float(elevation.mean()), elevation_std=float(elevation.std(unbiased=False)),
                                   mean_slope=float(slope.mean()), slope_std=float(slope.std(unbiased=False)))
                sample_rows.extend(new_rows)
            if logger is not None and (batch_index == 1 or batch_index % 100 == 0 or batch_index == len(val_loader)):
                logger.info("Full validation progress: %s/%s batches", batch_index, len(val_loader))

    metrics = overall.finalize()
    metrics["full_val_inference_time_seconds"] = timed_seconds
    metrics["full_val_inference_time_per_batch_ms"] = _safe_divide(1000.0 * timed_seconds, timed_batches)
    metrics["full_val_inference_time_per_sample_ms"] = _safe_divide(1000.0 * timed_seconds, timed_samples)
    metrics["full_val_samples_per_second"] = _safe_divide(timed_samples, timed_seconds)
    metrics["full_val_timing_warmup_batch_count"] = warmup_batches
    metrics["full_val_timing_batch_count"] = timed_batches
    metrics["full_val_timing_sample_count"] = timed_samples
    evaluated = int(metrics["full_val_total_patch_count"])
    if evaluated != expected_samples:
        raise RuntimeError(f"Full validation evaluated {evaluated} samples, expected exactly {expected_samples}.")
    if sample_id_count and (sample_id_count != evaluated or len(sample_ids) != evaluated):
        raise RuntimeError(
            f"Full-validation sample-ID coverage mismatch: ids={len(sample_ids)} metadata={sample_id_count} evaluated={evaluated}."
        )
    if save_analysis_data and len(sample_rows) != evaluated:
        raise RuntimeError(f"Sample-level metric row count {len(sample_rows)} does not equal full-validation count {evaluated}.")
    if save_analysis_data and len({row["sample_id"] for row in sample_rows}) != evaluated:
        raise RuntimeError("Sample-level metric rows do not contain one unique sample ID per validation patch.")
    if int(metrics["full_val_fire_patch_count"]) + int(metrics["full_val_no_fire_patch_count"]) != evaluated:
        raise RuntimeError("Full validation fire/no-fire counts do not equal the total.")
    if expected_counts is not None:
        expected_triplet = (
            int(expected_counts["total"]),
            int(expected_counts["fire"]),
            int(expected_counts["no_fire"]),
        )
        actual_triplet = (
            evaluated,
            int(metrics["full_val_fire_patch_count"]),
            int(metrics["full_val_no_fire_patch_count"]),
        )
        if expected_triplet[2] > 0 and actual_triplet[2] == 0:
            raise RuntimeError("Canonical validation scan found no-fire samples, but full evaluation counted zero.")
        if actual_triplet != expected_triplet:
            raise RuntimeError(f"Full-validation counts {actual_triplet} disagree with canonical counts {expected_triplet}.")

    per_fire_payload = {name: _per_fire_metrics(accumulator) for name, accumulator in sorted(per_fire.items())}
    if per_fire_payload and sum(int(item["sample_count"]) for item in per_fire_payload.values()) != evaluated:
        raise RuntimeError("Per-fire sample counts do not sum to the full-validation total.")
    activity_payload = {name: _activity_bin_metrics(per_activity[name]) for name in ACTIVE_FRACTION_BIN_NAMES}
    if sum(int(item["patch_count"]) for item in activity_payload.values()) != evaluated:
        raise RuntimeError("Activity-bin sample counts do not sum to the full-validation total.")
    run_path = Path(run_dir).expanduser().resolve()
    evaluation_path = run_path / "evaluation"
    created_at = datetime.now(timezone.utc).isoformat()
    full_payload = {
        "schema_version": 2,
        "created_at": created_at,
        "metric_scope": "full_validation_best_checkpoint",
        "split": "val",
        "model_name": model_name,
        "seed": seed,
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "classification": {
            "source": "ground_truth_future_target_mask_only",
            "mask_channel": FIRE_MASK_CHANNEL,
            "fire_pixel_rule": f"target_mask > {FIRE_MASK_THRESHOLD}",
            "active_fraction_threshold": ACTIVE_FRACTION_THRESHOLD,
            "prediction_threshold": PREDICTED_FIRE_THRESHOLD,
        },
        "dataset_sample_count": expected_samples,
        "evaluated_sample_count": evaluated,
        "unique_sample_id_count": len(sample_ids) if sample_id_count else None,
        "canonical_balance_counts": dict(expected_counts) if expected_counts is not None else None,
        "metrics": metrics,
    }
    per_fire_file_payload = {
        "schema_version": 2,
        "created_at": created_at,
        "metric_scope": "full_validation_per_fire_best_checkpoint",
        "model_name": model_name,
        "seed": seed,
        "fires": per_fire_payload,
    }
    activity_file_payload = {
        "schema_version": 2,
        "created_at": created_at,
        "metric_scope": "full_validation_by_activity_bin_best_checkpoint",
        "model_name": model_name,
        "seed": seed,
        "classification": {
            "source": "ground_truth_future_target_mask_only",
            "bin_names": list(ACTIVE_FRACTION_BIN_NAMES),
            "boundaries": list(ACTIVE_FRACTION_BOUNDARIES),
        },
        "bins": activity_payload,
    }
    _atomic_json(run_path / "full_validation_metrics.json", full_payload)
    _atomic_json(run_path / "full_validation_per_fire.json", per_fire_file_payload)
    _atomic_json(run_path / "full_validation_by_activity_bin.json", activity_file_payload)

    per_fire_rows: list[dict[str, Any]] = []
    for fire_name, values in per_fire_payload.items():
        per_fire_rows.append(
            {
                "model_name": model_name,
                "seed": seed,
                "fire_name": fire_name,
                "sample_count": values["sample_count"],
                "fire_patch_count": values["fire_patch_count"],
                "no_fire_patch_count": values["no_fire_patch_count"],
                "dice": values["dice"],
                "iou": values["iou"],
                "precision": values["precision"],
                "recall": values["recall"],
                "surface_mae": values["surface_mae"],
                "surface_rmse": values["surface_rmse"],
                "canopy_mae": values["canopy_mae"],
                "canopy_rmse": values["canopy_rmse"],
                "energy_log_mae": values["energy_log_mae"],
                "energy_log_rmse": values["energy_log_rmse"],
                "active_canopy_mae": values["active_canopy_mae"],
                "active_energy_log_mae": values["active_energy_log_mae"],
                "no_fire_pixel_fp_rate": values["no_fire_mask_false_positive_rate"],
                "no_fire_patch_fp_rate": values["no_fire_patch_false_positive_rate"],
                "no_fire_surface_abs_mean": values["no_fire_surface_abs_pred_mean"],
                "no_fire_canopy_abs_mean": values["no_fire_canopy_abs_pred_mean"],
                "no_fire_energy_log_abs_mean": values["no_fire_energy_log_abs_pred_mean"],
            }
        )
    per_fire_csv_path = evaluation_path / "per_fire_validation_metrics.csv"
    _atomic_csv(
        per_fire_csv_path,
        per_fire_rows,
        fieldnames=(
            "model_name", "seed", "fire_name", "sample_count", "fire_patch_count", "no_fire_patch_count",
            "dice", "iou", "precision", "recall", "surface_mae", "surface_rmse", "canopy_mae", "canopy_rmse",
            "energy_log_mae", "energy_log_rmse", "active_canopy_mae", "active_energy_log_mae",
            "no_fire_pixel_fp_rate", "no_fire_patch_fp_rate", "no_fire_surface_abs_mean",
            "no_fire_canopy_abs_mean", "no_fire_energy_log_abs_mean",
        ),
    )

    sample_metrics_path: Path | None = None
    qualitative_predictions_path: Path | None = None
    if save_analysis_data:
        # CSV is the explicit dependency-free fallback allowed by the paper-data specification.
        sample_metrics_path = evaluation_path / "validation_sample_metrics.csv"
        _atomic_csv(sample_metrics_path, sample_rows)
        selected_order = [str(item["sample_id"]) for item in (qualitative_payload or {}).get("selected_samples", [])]
        missing_qualitative = [sample_id for sample_id in selected_order if sample_id not in qualitative_arrays]
        if missing_qualitative:
            raise RuntimeError(f"Full validation did not visit qualitative sample IDs: {missing_qualitative[:10]}")
        qualitative_predictions_path = evaluation_path / "qualitative_predictions.npz"
        qualitative_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_npz = qualitative_predictions_path.with_name(qualitative_predictions_path.name + ".tmp.npz")
        ordered = [qualitative_arrays[sample_id] for sample_id in selected_order]
        np.savez_compressed(
            temporary_npz,
            sample_id=np.asarray([item["sample_id"] for item in ordered]),
            fire_name=np.asarray([item["fire_name"] for item in ordered]),
            activity_bin=np.asarray([item["activity_bin"] for item in ordered]),
            target_surface=np.stack([item["target_surface"] for item in ordered]),
            target_canopy=np.stack([item["target_canopy"] for item in ordered]),
            target_mask=np.stack([item["target_mask"] for item in ordered]),
            target_energy_log=np.stack([item["target_energy_log"] for item in ordered]),
            pred_surface=np.stack([item["pred_surface"] for item in ordered]),
            pred_canopy=np.stack([item["pred_canopy"] for item in ordered]),
            pred_mask_probability=np.stack([item["pred_mask_probability"] for item in ordered]),
            pred_energy_log=np.stack([item["pred_energy_log"] for item in ordered]),
        )
        temporary_npz.replace(qualitative_predictions_path)

    return {
        "metrics": metrics,
        "per_fire": per_fire_payload,
        "by_activity_bin": activity_payload,
        "artifact": full_payload,
        "per_fire_metrics_path": str(per_fire_csv_path),
        "sample_metrics_path": None if sample_metrics_path is None else str(sample_metrics_path),
        "qualitative_predictions_path": None if qualitative_predictions_path is None else str(qualitative_predictions_path),
    }


__all__ = ["FullValidationAccumulator", "evaluate_full_validation"]
