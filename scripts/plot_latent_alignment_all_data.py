#!/usr/bin/env python3
"""Create the FLARE 2x2 latent-alignment figure from every dataset fire.

Unlike the main-paper held-out-test analysis, this descriptive visualization
draws a deterministic activity-balanced sample from every fire in the train,
validation, and test splits. Every selected sample is projected and included
in the original-space latent diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders  # noqa: E402
from src.evaluation.fire_activity import (  # noqa: E402
    ACTIVE_FRACTION_BIN_NAMES,
    ACTIVE_FRACTION_BOUNDARIES,
    FIRE_MASK_THRESHOLD,
    active_fraction_bin_name,
    classify_fire_masks,
)
from src.evaluation.validation_subset import classify_validation_records  # noqa: E402
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.batch_utils import unpack_batch  # noqa: E402
from src.training.checkpoints import (  # noqa: E402
    load_checkpoint,
    load_model_state_dict_compatible,
    validate_checkpoint_model_compatibility,
)
from src.training.input_normalization import (  # noqa: E402
    apply_input_normalization,
    build_input_normalizer_for_loader,
)

from plot_latent_alignment import (  # noqa: E402
    ACTIVITY_DISPLAY_NAMES,
    ALLOWED_MODEL_SEEDS,
    METRIC_FIELDS,
    PAPER_NAMES,
    _atomic_json,
    _atomic_npz,
    _format_metric,
    _git_commit,
    _protocol_signature,
    _resolve_table2_run,
    _run_record,
    _training_mmd_metadata,
    _write_metrics_csv,
    draw_latent_figure,
    latent_diagnostics,
    select_balanced_samples,
    sha256_file,
)


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/latent_alignment"
SPLIT_ORDER = ("train", "val", "test")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _AllRecordsView:
    """Target-classification view over concatenated processed split records."""

    def __init__(self, datasets: Sequence[Any]):
        roots = {str(Path(dataset.root).resolve()) for dataset in datasets}
        if len(roots) != 1:
            raise RuntimeError(f"All split datasets must share one processed root, got {sorted(roots)}")
        self.root = Path(next(iter(roots)))
        self.records = [record for dataset in datasets for record in dataset.records]

    def __len__(self) -> int:
        return len(self.records)


def _runtime_loaders(config: Mapping[str, Any], batch_size: int, num_workers: int, pin_memory: bool):
    runtime = copy.deepcopy(dict(config))
    runtime["return_metadata"] = True
    dataloader = dict(runtime.get("dataloader", {}))
    dataloader["return_metadata"] = True
    dataloader["include_test_split"] = True
    runtime["dataloader"] = dataloader
    data_loader = copy.deepcopy(dict(runtime.get("data_loader", {})))
    for split in SPLIT_ORDER:
        options = dict(data_loader.get(split, {}))
        options.update(
            {
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "persistent_workers": bool(num_workers > 0),
                "pin_memory": bool(pin_memory),
                "drop_last": False,
            }
        )
        data_loader[split] = options
    data_loader["batch_size"] = int(batch_size)
    runtime["data_loader"] = data_loader
    loaders = create_dataloaders(runtime)
    if any(loader is None for loader in loaders):
        raise RuntimeError("The frozen configuration did not construct all train/val/test datasets.")
    by_split = dict(zip(SPLIT_ORDER, loaders))
    for split, loader in by_split.items():
        if str(getattr(loader.dataset, "split", "")) != split:
            raise RuntimeError(f"Expected {split!r} dataset, got {getattr(loader.dataset, 'split', None)!r}.")
    return runtime, by_split


def _validate_complete_fire_inventory(config: Mapping[str, Any], loaders: Mapping[str, Any]) -> dict[str, list[str]]:
    section = config.get("manual_fire_split", {})
    inventory: dict[str, list[str]] = {}
    all_observed: set[str] = set()
    for split in SPLIT_ORDER:
        expected = {str(value) for value in section.get(f"{split}_fires", [])}
        observed = {str(record["fire_name"]) for record in loaders[split].dataset.records}
        if not expected or observed != expected:
            raise RuntimeError(
                f"Complete {split} fire inventory mismatch: expected={sorted(expected)} observed={sorted(observed)}"
            )
        if all_observed & observed:
            raise RuntimeError(f"Fire identity leaks across splits: {sorted(all_observed & observed)}")
        all_observed.update(observed)
        inventory[split] = sorted(observed)
    return inventory


def _build_all_selection(
    loaders: Mapping[str, Any],
    signature: Mapping[str, Any],
    fire_inventory: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    datasets = [loaders[split].dataset for split in SPLIT_ORDER]
    view = _AllRecordsView(datasets)
    print(f"CLASSIFYING CANONICAL ACTIVITY BINS: 0/{len(view)}", flush=True)
    classified = classify_validation_records(view, progress_every=0)
    if len(classified) != len(view):
        raise RuntimeError("Canonical activity classification did not cover the complete dataset.")
    split_offsets: dict[str, int] = {}
    offset = 0
    for split, dataset in zip(SPLIT_ORDER, datasets):
        split_offsets[split] = offset
        offset += len(dataset)

    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(classified):
        record = view.records[index]
        sample_id = str(item["sample_id"])
        split = str(record["split"])
        if split not in SPLIT_ORDER:
            raise RuntimeError(f"Unexpected split {split!r} for {sample_id}")
        if sample_id in seen_ids:
            raise RuntimeError(f"Duplicate sample ID across complete dataset: {sample_id}")
        seen_ids.add(sample_id)
        activity_bin = str(item["activity_bin"])
        if activity_bin not in ACTIVE_FRACTION_BIN_NAMES:
            raise RuntimeError(f"Noncanonical activity bin {activity_bin!r}")
        samples.append(
            {
                "selection_rank": index,
                "dataset_index": index,
                "split_index": index - split_offsets[split],
                "sample_id": sample_id,
                "split": split,
                "fire_id": str(item["fire_name"]),
                "activity_bin": activity_bin,
                "active_fraction": float(item["active_fraction"]),
                "active_pixel_count": int(item["active_pixels"]),
            }
        )
        completed = index + 1
        if completed % 25000 == 0 or completed == len(classified):
            print(f"CLASSIFYING CANONICAL ACTIVITY BINS: {completed}/{len(classified)}", flush=True)

    split_counts = Counter(str(item["split"]) for item in samples)
    fire_counts = Counter(str(item["fire_id"]) for item in samples)
    expected_total = sum(len(loaders[split].dataset) for split in SPLIT_ORDER)
    if len(samples) != expected_total:
        raise RuntimeError(f"Complete selection count mismatch: {len(samples)} != {expected_total}")
    for split, fires in fire_inventory.items():
        allowed = set(fires)
        if any(item["fire_id"] not in allowed for item in samples if item["split"] == split):
            raise RuntimeError(f"Selection contains a {split} sample assigned to another split's fire.")
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "data_scope": "complete_train_validation_test_dataset",
        "splits": list(SPLIT_ORDER),
        "selection_algorithm": "all_samples_no_subsampling_v1",
        "selection_uses_predictions_or_features": False,
        "selection_completed_before_checkpoint_loading": True,
        "selection_basis": "complete_frozen_sample_index_and_ground_truth_target_mask_only",
        "fire_mask_threshold": FIRE_MASK_THRESHOLD,
        "canonical_activity_bin_names": list(ACTIVE_FRACTION_BIN_NAMES),
        "canonical_activity_bin_boundaries": list(ACTIVE_FRACTION_BOUNDARIES),
        "sample_index_path": signature["sample_index_path"],
        "sample_index_sha256": signature["sample_index_sha256"],
        "dataset_sample_count": len(samples),
        "selected_sample_count": len(samples),
        "all_samples_selected": True,
        "split_counts": {split: int(split_counts[split]) for split in SPLIT_ORDER},
        "fire_inventory_by_split": {split: list(fires) for split, fires in fire_inventory.items()},
        "fire_counts": dict(sorted(fire_counts.items())),
        "activity_bin_counts": {
            name: int(sum(item["activity_bin"] == name for item in samples))
            for name in ACTIVE_FRACTION_BIN_NAMES
        },
        "samples": samples,
    }


def _balanced_selection_from_population(
    population: Mapping[str, Any],
    *,
    selection_seed: int,
    samples_per_fire: int,
) -> dict[str, Any]:
    """Select the same deterministic fire/activity-balanced subset used by the test figure."""

    population_samples = population["samples"]
    classified = [
        {
            "index": int(item["dataset_index"]),
            "sample_id": str(item["sample_id"]),
            "fire_name": str(item["fire_id"]),
            "activity_bin": str(item["activity_bin"]),
            "active_fraction": float(item["active_fraction"]),
            "active_pixels": int(item["active_pixel_count"]),
        }
        for item in population_samples
    ]
    selected, audit = select_balanced_samples(
        classified,
        selection_seed=int(selection_seed),
        samples_per_fire=int(samples_per_fire),
    )
    source_by_index = {int(item["dataset_index"]): item for item in population_samples}
    samples: list[dict[str, Any]] = []
    for rank, item in enumerate(selected):
        source = source_by_index[int(item["index"])]
        samples.append(
            {
                "selection_rank": rank,
                "dataset_index": int(source["dataset_index"]),
                "split_index": int(source["split_index"]),
                "sample_id": str(source["sample_id"]),
                "split": str(source["split"]),
                "fire_id": str(source["fire_id"]),
                "activity_bin": str(source["activity_bin"]),
                "active_fraction": float(source["active_fraction"]),
                "active_pixel_count": int(source["active_pixel_count"]),
            }
        )
    fire_counts = Counter(str(item["fire_id"]) for item in samples)
    split_counts = Counter(str(item["split"]) for item in samples)
    activity_counts = Counter(str(item["activity_bin"]) for item in samples)
    population_fire_count = len(population.get("fire_counts", {}))
    if len(fire_counts) != population_fire_count:
        raise RuntimeError(
            f"Balanced selection omitted fires: selected={len(fire_counts)} population={population_fire_count}"
        )
    return {
        "schema_version": 2,
        "created_at": _utc_now(),
        "data_scope": "balanced_sample_from_complete_train_validation_test_fire_inventory",
        "splits": list(SPLIT_ORDER),
        "selection_algorithm": audit["algorithm"],
        "selection_uses_predictions_or_features": False,
        "selection_completed_before_checkpoint_loading": True,
        "selection_basis": "complete_frozen_sample_index_and_ground_truth_target_mask_only",
        "fire_mask_threshold": population["fire_mask_threshold"],
        "canonical_activity_bin_names": list(ACTIVE_FRACTION_BIN_NAMES),
        "canonical_activity_bin_boundaries": list(ACTIVE_FRACTION_BOUNDARIES),
        "sample_index_path": population["sample_index_path"],
        "sample_index_sha256": population["sample_index_sha256"],
        "population_sample_count": int(population["dataset_sample_count"]),
        "population_split_counts": dict(population["split_counts"]),
        "population_fire_counts": dict(population["fire_counts"]),
        "selected_sample_count": len(samples),
        "all_samples_selected": False,
        "all_fires_included": True,
        "selection_seed": int(selection_seed),
        "samples_per_fire_requested": int(samples_per_fire),
        "split_counts": {split: int(split_counts[split]) for split in SPLIT_ORDER},
        "fire_inventory_by_split": dict(population["fire_inventory_by_split"]),
        "fire_counts": dict(sorted(fire_counts.items())),
        "activity_bin_counts": {
            name: int(activity_counts[name]) for name in ACTIVE_FRACTION_BIN_NAMES
        },
        "audit": audit,
        "samples": samples,
    }


def _load_model(run: Mapping[str, Any], device: torch.device):
    config = run["config"]
    model = build_model_from_config(config, input_channels=int(config["model"]["input_channels"])).to(device)
    checkpoint = load_checkpoint(run["checkpoint"], map_location="cpu")
    validate_checkpoint_model_compatibility(model, checkpoint, run["checkpoint"])
    load_model_state_dict_compatible(model, checkpoint, run["checkpoint"])
    model.eval()
    epoch = checkpoint.get("epoch")
    del checkpoint
    return model, epoch


def _collated_sample_ids(metadata: Any) -> list[str]:
    if not isinstance(metadata, Mapping) or "sample_id" not in metadata:
        raise RuntimeError("All-data inference batch lacks sample_id metadata.")
    value = metadata["sample_id"]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if torch.is_tensor(value):
        return [str(item) for item in value.tolist()]
    return [str(value)]


def _extract_all_latents(
    *,
    left_run: Mapping[str, Any],
    final_run: Mapping[str, Any],
    loaders: Mapping[str, Any],
    selection: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    datasets = [loaders[split].dataset for split in SPLIT_ORDER]
    combined = ConcatDataset(datasets)
    selected = selection["samples"]
    selected_indices = [int(item["dataset_index"]) for item in selected]
    if not selected_indices or min(selected_indices) < 0 or max(selected_indices) >= len(combined):
        raise RuntimeError("Balanced selection contains an invalid complete-dataset index.")
    loader = DataLoader(
        Subset(combined, selected_indices),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        persistent_workers=bool(num_workers > 0),
        pin_memory=device.type == "cuda",
    )
    normalizer = build_input_normalizer_for_loader(
        loaders["test"], device, int(left_run["config"]["model"]["input_channels"]), left_run["config"]
    )
    left_model, left_checkpoint_epoch = _load_model(left_run, device)
    final_model, final_checkpoint_epoch = _load_model(final_run, device)
    left_features: list[np.ndarray] = []
    final_features: list[np.ndarray] = []
    position = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            x, y, extra = unpack_batch(batch)
            count = int(x.shape[0])
            expected = selected[position : position + count]
            expected_ids = [str(item["sample_id"]) for item in expected]
            actual_ids = _collated_sample_ids(extra.get("metadata"))
            if actual_ids != expected_ids:
                raise RuntimeError(f"All-data inference order changed at position {position}.")
            classified = classify_fire_masks(y[:, 2])
            fractions = [float(value) for value in classified["active_fraction"].tolist()]
            bins = [active_fraction_bin_name(value) for value in fractions]
            if bins != [str(item["activity_bin"]) for item in expected]:
                raise RuntimeError("Canonical activity labels changed between preselection and inference.")
            x = x.float().to(device, non_blocking=device.type == "cuda")
            x = apply_input_normalization(x, normalizer, left_run["config"])
            terrain = extra.get("terrain")
            if torch.is_tensor(terrain):
                terrain = terrain.float().to(device, non_blocking=device.type == "cuda")
            else:
                terrain = None
            pooled: list[np.ndarray] = []
            for model in (left_model, final_model):
                output = (
                    model(x, return_features=True)
                    if terrain is None
                    else model(x, terrain=terrain, return_features=True)
                )
                if not isinstance(output, Mapping) or "z_shared" not in output:
                    raise RuntimeError("FLARE return_features=True did not return canonical z_shared.")
                latent = output["z_shared"]
                if latent.ndim != 4 or int(latent.shape[0]) != count or not torch.isfinite(latent).all():
                    raise RuntimeError(f"Invalid z_shared tensor: shape={tuple(latent.shape)}")
                pooled.append(latent.mean(dim=(-2, -1)).float().cpu().numpy())
                del output, latent
            left_features.append(pooled[0])
            final_features.append(pooled[1])
            position += count
            if (batch_index + 1) % 50 == 0 or position == len(selected):
                print(f"ALL-DATA LATENT EXTRACTION: {position}/{len(selected)}", flush=True)
    left_array = np.concatenate(left_features, axis=0).astype(np.float32, copy=False)
    final_array = np.concatenate(final_features, axis=0).astype(np.float32, copy=False)
    if left_array.shape[0] != len(selected) or final_array.shape[0] != len(selected):
        raise RuntimeError("All-data extraction did not return one latent per sample.")
    if left_array.shape[1] != final_array.shape[1]:
        raise RuntimeError(f"Shared latent dimensions differ: {left_array.shape} vs {final_array.shape}")
    return left_array, final_array, {
        "left_checkpoint_epoch": left_checkpoint_epoch,
        "final_checkpoint_epoch": final_checkpoint_epoch,
        "latent_key": "z_shared",
        "latent_tensor_layout": "B,D,H,W",
        "pooling": "unweighted_global_average_over_H_W",
        "pooled_dimension": int(left_array.shape[1]),
        "device": str(device),
        "model_mode": "eval",
        "autograd": False,
        "same_tensor_batches_supplied_to_both_models": True,
        "complete_dataset_extracted": False,
        "balanced_all_fire_selection_extracted": True,
    }


def _diagnostic_indices(selection: Mapping[str, Any], maximum: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    samples = selection["samples"]
    if int(maximum) <= 0 or int(maximum) >= len(samples):
        indices = np.arange(len(samples), dtype=np.int64)
        return indices, {
            "algorithm": "all_selected_samples_v1",
            "selected_population_sample_count": len(samples),
            "requested_maximum": int(maximum),
            "actual_diagnostic_sample_count": len(samples),
            "diagnostic_subset_only": False,
        }
    fire_count = len({str(item["fire_id"]) for item in samples})
    per_fire = max(1, int(maximum) // fire_count)
    classified = [
        {
            "index": int(item["dataset_index"]),
            "sample_id": str(item["sample_id"]),
            "fire_name": str(item["fire_id"]),
            "activity_bin": str(item["activity_bin"]),
        }
        for item in samples
    ]
    chosen, audit = select_balanced_samples(
        classified, selection_seed=int(seed), samples_per_fire=per_fire
    )
    indices = np.asarray([int(item["index"]) for item in chosen], dtype=np.int64)
    audit.update(
        {
            "population_sample_count": len(samples),
            "requested_maximum": int(maximum),
            "actual_diagnostic_sample_count": int(indices.size),
            "diagnostic_subset_only": True,
        }
    )
    return indices, audit


def _diagnostics(
    features: np.ndarray,
    selection: Mapping[str, Any],
    indices: np.ndarray,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    samples = selection["samples"]
    fire = [str(samples[index]["fire_id"]) for index in indices]
    activity = [str(samples[index]["activity_bin"]) for index in indices]
    result = latent_diagnostics(features[indices], fire, activity)
    result["population_sample_count"] = len(samples)
    result["diagnostics_are_deterministic_subset_estimates"] = True
    result["diagnostic_sampling"] = dict(audit)
    return result


def _project_all(
    left_features: np.ndarray,
    final_features: np.ndarray,
    *,
    requested: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fallback_reason = None
    if requested == "umap":
        try:
            import umap  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "UMAP was requested for the all-fire figure, but umap-learn is not installed. "
                "Install umap-learn in the fire_forecasting environment; this script will not silently use PCA."
            ) from exc
        else:
            neighbors = min(max(2, int(n_neighbors)), int(left_features.shape[0]) - 1)

            def fit_umap(values: np.ndarray) -> np.ndarray:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=neighbors,
                    min_dist=float(min_dist),
                    metric="euclidean",
                    random_state=int(seed),
                    transform_seed=int(seed),
                    n_jobs=1,
                    low_memory=True,
                )
                return np.asarray(reducer.fit_transform(values), dtype=np.float32)

            return fit_umap(left_features), fit_umap(final_features), {
                "requested": "umap",
                "effective": "umap",
                "fit_separately_per_model": True,
                "projection_seed": int(seed),
                "n_neighbors": neighbors,
                "min_dist": float(min_dist),
            "all_selected_samples_projected": True,
                "fallback_reason": None,
            }

    details: dict[str, Any] = {}
    coordinates: list[np.ndarray] = []
    for key, values in (("baseline", left_features), ("final", final_features)):
        reducer = PCA(n_components=2, svd_solver="randomized", random_state=int(seed))
        coordinates.append(reducer.fit_transform(values).astype(np.float32, copy=False))
        details[key] = {
            "method": "randomized_pca",
            "explained_variance_ratio": [float(value) for value in reducer.explained_variance_ratio_],
        }
    return coordinates[0], coordinates[1], {
        "requested": requested,
        "effective": "pca",
        "fit_separately_per_model": True,
        "projection_seed": int(seed),
        "all_selected_samples_projected": True,
        "fallback_reason": fallback_reason,
        **details,
    }


def _all_fire_display_names(fire_ids: Sequence[str]) -> dict[str, str]:
    simple = {
        "ARCHIECREEK": "Archie Creek",
        "CALWOOD": "Calwood",
        "CAMP": "Camp",
        "CARR_FIRENADO": "Carr Firenado",
        "CHIMNEYTOPS2": "Chimney Tops 2",
        "CREEK": "Creek",
        "EASTTROUBLESOME": "East Troublesome",
        "ESPERANZA": "Esperanza",
        "GLASS": "Glass",
        "MARSHALL": "Marshall",
        "MCKINNEY": "McKinney",
        "MOSQUITO": "Mosquito",
        "NORTHCOMPLX": "North Complex",
        "PARKCTY_HYPOTHETICAL": "Park City (hyp.)",
        "REDWOODVALLEY": "Redwood Valley",
        "SLATER": "Slater",
        "SPRINGS": "Springs",
        "TRUCKEE_HYPOTHETICAL": "Truckee (hyp.)",
        "TUOLUMNE_65": "Tuolumne 65",
        "Tubbs": "Tubbs",
        "WITCHCREEK": "Witch Creek",
        "WOOLSEY": "Woolsey",
        "YUBA_HYPOTHETICAL": "Yuba (hyp.)",
    }
    qualifier_names = {
        "THOMAS__1204__keepz_08": "Thomas (Dec. 4)",
        "THOMAS__1208__keepz_08": "Thomas (Dec. 8)",
        "THOMAS__1209__keepz_08": "Thomas (Dec. 9)",
        "THOMAS__1210__keepz_08": "Thomas (Dec. 10)",
        "KING__period1__keepz_08": "King (period 1)",
        "KING__period2__keepz_08": "King (period 2)",
        "KING__period3__keepz_08": "King (period 3)",
        "LITTLEBEAR__lbvaa__keepz_08": "Little Bear (A)",
        "LITTLEBEAR__lbvab__keepz_08": "Little Bear (B)",
        "LITTLEBEAR__lbvac__keepz_08": "Little Bear (C)",
        "RIM__0820__keepz_08": "Rim (Aug. 20)",
        "RIM__0822__keepz_08": "Rim (Aug. 22)",
    }
    labels = {name: qualifier_names.get(name, simple.get(name, name.replace("_", " ").title())) for name in fire_ids}
    if len(set(labels.values())) != len(labels):
        raise RuntimeError("Cleaned all-fire display names are not unique.")
    return labels


def _write_coordinates(
    path: Path,
    selection: Mapping[str, Any],
    left_coordinates: np.ndarray,
    final_coordinates: np.ndarray,
) -> None:
    fields = [
        "sample_id",
        "split",
        "dataset_index",
        "fire_id",
        "activity_bin",
        "active_fraction",
        "baseline_x",
        "baseline_y",
        "final_x",
        "final_y",
    ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(selection["samples"]):
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "split": sample["split"],
                    "dataset_index": sample["dataset_index"],
                    "fire_id": sample["fire_id"],
                    "activity_bin": sample["activity_bin"],
                    "active_fraction": repr(float(sample["active_fraction"])),
                    "baseline_x": repr(float(left_coordinates[index, 0])),
                    "baseline_y": repr(float(left_coordinates[index, 1])),
                    "final_x": repr(float(final_coordinates[index, 0])),
                    "final_y": repr(float(final_coordinates[index, 1])),
                }
            )
    temporary.replace(path)


def _save_latents(path: Path, selection: Mapping[str, Any], left: np.ndarray, final: np.ndarray) -> None:
    samples = selection["samples"]
    _atomic_npz(
        path,
        sample_ids=np.asarray([item["sample_id"] for item in samples], dtype="U"),
        splits=np.asarray([item["split"] for item in samples], dtype="U"),
        fire_ids=np.asarray([item["fire_id"] for item in samples], dtype="U"),
        activity_bins=np.asarray([item["activity_bin"] for item in samples], dtype="U"),
        active_fractions=np.asarray([item["active_fraction"] for item in samples], dtype=np.float64),
        dataset_indices=np.arange(len(samples), dtype=np.int64),
        baseline_features=np.asarray(left, dtype=np.float32),
        final_features=np.asarray(final, dtype=np.float32),
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this process.")
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        major, minor = torch.cuda.get_device_capability(index)
        required_arch = f"sm_{major}{minor}"
        compiled_arches = set(torch.cuda.get_arch_list())
        if required_arch not in compiled_arches:
            name = torch.cuda.get_device_name(index)
            raise RuntimeError(
                f"CUDA device {name} requires {required_arch}, but torch {torch.__version__} "
                f"was compiled for {sorted(compiled_arches)}. Choose a compatible GPU or use --device cpu."
            )
    return device


def run(args: argparse.Namespace) -> Path:
    output_dir = (
        args.output_dir
        or (args.output_root / f"all_fires_seed_{int(args.selection_seed):03d}")
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selected_samples.json"
    pooled_path = output_dir / "pooled_latents.npz"
    completed = [output_dir / name for name in ("latent_alignment.pdf", "latent_alignment.svg", "latent_alignment.png", "pooled_latents.npz") if (output_dir / name).exists()]
    if completed and not args.overwrite:
        raise FileExistsError(f"All-data outputs already exist: {completed}. Use --overwrite explicitly.")

    final_run = _resolve_table2_run("final", args.model_seed)
    baseline_run = _resolve_table2_run("baseline", args.model_seed)
    baseline_signature = _protocol_signature(baseline_run)
    final_signature = _protocol_signature(final_run)
    if baseline_signature != final_signature:
        raise RuntimeError("FLARE baseline and final do not share exact preprocessing/normalization identity.")
    device = _device(args.device)
    runtime, loaders = _runtime_loaders(
        baseline_run["config"], args.batch_size, args.num_workers, device.type == "cuda"
    )
    fire_inventory = _validate_complete_fire_inventory(runtime, loaders)
    expected_split_counts = {split: len(loaders[split].dataset) for split in SPLIT_ORDER}
    expected_total = sum(expected_split_counts.values())
    if selection_path.is_file() and not args.reselect:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection.get("sample_index_sha256") != baseline_signature["sample_index_sha256"]
            or int(selection.get("population_sample_count", -1)) != expected_total
            or selection.get("all_fires_included") is not True
            or int(selection.get("selection_seed", -1)) != int(args.selection_seed)
            or int(selection.get("samples_per_fire_requested", -1)) != int(args.samples_per_fire)
        ):
            raise RuntimeError("Existing all-fire balanced selection is incompatible; use --reselect.")
    else:
        if args.reselect and pooled_path.exists():
            raise RuntimeError("Cannot reselect while pooled features exist; use a new output directory.")
        population_path = args.population_manifest.expanduser().resolve()
        if population_path.is_file():
            population = json.loads(population_path.read_text(encoding="utf-8"))
            if (
                population.get("sample_index_sha256") != baseline_signature["sample_index_sha256"]
                or int(population.get("selected_sample_count", -1)) != expected_total
                or population.get("all_samples_selected") is not True
            ):
                raise RuntimeError(f"Existing population manifest is incompatible: {population_path}")
        else:
            population = _build_all_selection(loaders, baseline_signature, fire_inventory)
        selection = _balanced_selection_from_population(
            population,
            selection_seed=args.selection_seed,
            samples_per_fire=args.samples_per_fire,
        )
        del population
        _atomic_json(selection_path, selection)
    if args.selection_only:
        print(f"Balanced all-fire selection written: {selection_path}\nSamples: {selection['selected_sample_count']}\nFires: {len(selection['fire_counts'])}")
        return output_dir

    left_features, final_features, extraction = _extract_all_latents(
        left_run=baseline_run,
        final_run=final_run,
        loaders=loaders,
        selection=selection,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    _save_latents(pooled_path, selection, left_features, final_features)

    diagnostic_indices, diagnostic_audit = _diagnostic_indices(
        selection, args.diagnostic_samples, args.diagnostic_seed
    )
    baseline_metrics = _diagnostics(left_features, selection, diagnostic_indices, diagnostic_audit)
    final_metrics = _diagnostics(final_features, selection, diagnostic_indices, diagnostic_audit)
    left_coordinates, final_coordinates, projection = _project_all(
        left_features,
        final_features,
        requested=args.projection,
        seed=args.projection_seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    fire_ids = [str(item["fire_id"]) for item in selection["samples"]]
    activity_bins = [str(item["activity_bin"]) for item in selection["samples"]]
    figure_paths = draw_latent_figure(
        output_dir=output_dir,
        left_coordinates=left_coordinates,
        final_coordinates=final_coordinates,
        fire_ids=fire_ids,
        activity_bins=activity_bins,
        left_metrics=baseline_metrics,
        final_metrics=final_metrics,
        left_title=PAPER_NAMES["baseline"],
        annotate_metrics=bool(args.annotate_metrics),
        fire_display_names=_all_fire_display_names(sorted(set(fire_ids))),
        large_dataset_style=True,
    )
    _write_coordinates(output_dir / "embedding_coordinates.csv", selection, left_coordinates, final_coordinates)

    metrics = {"baseline": baseline_metrics, "final": final_metrics}
    metrics_payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "data_scope": "balanced_sample_from_complete_train_validation_test_fire_inventory",
        "plotted_sample_count": len(selection["samples"]),
        "all_selected_samples_projected_and_plotted": True,
        "diagnostic_space": "original_high_dimensional_global_average_pooled_z_shared",
        "diagnostics_use_projection": False,
        "diagnostics_are_deterministic_subset_estimates": True,
        "diagnostic_sampling": diagnostic_audit,
        "models": metrics,
        "comparison": {
            f"final_minus_baseline_{field}": None
            if baseline_metrics.get(field) is None or final_metrics.get(field) is None
            else float(final_metrics[field]) - float(baseline_metrics[field])
            for field in METRIC_FIELDS
        },
        "mmd_training_metadata": _training_mmd_metadata(final_run),
    }
    _atomic_json(output_dir / "latent_metrics.json", metrics_payload)
    _write_metrics_csv(
        output_dir / "latent_metrics.csv",
        metrics,
        {"baseline": PAPER_NAMES["baseline"], "final": PAPER_NAMES["final"]},
    )
    _atomic_json(output_dir / "diagnostic_samples.json", {
        "schema_version": 1,
        "audit": diagnostic_audit,
        "dataset_indices": diagnostic_indices.tolist(),
        "sample_ids": [selection["samples"][int(index)]["sample_id"] for index in diagnostic_indices],
    })

    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "terminology": {"CAWFE-Latte": "FLARE", "GA_Q2": "FLARE final", "baseline": "FLARE baseline"},
        "data_scope": "balanced_sample_from_complete_train_validation_test_fire_inventory",
        "descriptive_not_held_out_generalization_analysis": True,
        "model_seed": int(args.model_seed),
        "selection_seed": int(args.selection_seed),
        "samples_per_fire_requested": int(args.samples_per_fire),
        "models": {"baseline": _run_record(baseline_run), "final": _run_record(final_run)},
        "latent_feature": extraction,
        "population_sample_count": int(selection["population_sample_count"]),
        "selected_sample_count": len(selection["samples"]),
        "split_counts": selection["split_counts"],
        "fire_count": len(selection["fire_counts"]),
        "fire_counts": selection["fire_counts"],
        "activity_bin_counts": selection["activity_bin_counts"],
        "canonical_activity_bins": list(ACTIVE_FRACTION_BIN_NAMES),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "projection": projection,
        "all_selected_samples_projected": True,
        "projection_fit_separately_per_model": True,
        "diagnostics_computed_in_original_latent_space": True,
        "diagnostic_sampling": diagnostic_audit,
        "scatter_layers_rasterized_in_vector_outputs": True,
        "candidate_ranking": "none",
        "ga_control_available_in_repository": False,
        "attribution_caveat": "FLARE baseline and FLARE final differ in backbone/decoder architecture as well as MMD; this comparison cannot isolate MMD.",
        "scope_caveat": "This figure includes training and validation samples and is descriptive; it is not evidence of held-out generalization.",
        "outputs": {
            **figure_paths,
            "selected_samples": str(selection_path),
            "pooled_latents": str(pooled_path),
            "embedding_coordinates": str(output_dir / "embedding_coordinates.csv"),
            "latent_metrics_json": str(output_dir / "latent_metrics.json"),
            "latent_metrics_csv": str(output_dir / "latent_metrics.csv"),
            "diagnostic_samples": str(output_dir / "diagnostic_samples.json"),
        },
        "git_commit": _git_commit(),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__},
    }
    _atomic_json(output_dir / "figure_metadata.json", metadata)

    print("\nLATENT ALIGNMENT ANALYSIS — ALL FIRES")
    print(f"\nFLARE baseline checkpoint:\n  {baseline_run['checkpoint']}")
    print(f"\nFLARE final checkpoint:\n  {final_run['checkpoint']}")
    print(f"\nModel seed:\n  {args.model_seed}")
    print(f"\nDataset population:\n  {selection['population_sample_count']}")
    print(f"\nBalanced samples plotted:\n  {len(selection['samples'])}")
    print(f"\nFire IDs:\n  {len(selection['fire_counts'])}")
    print(f"\nSplit counts:\n  {selection['split_counts']}")
    print(f"\nLatent feature:\n  Z_shared -> global average pooled")
    print(f"\nProjection:\n  {projection['effective'].upper()}")
    if projection.get("fallback_reason"):
        print(f"  FALLBACK: {projection['fallback_reason']}")
    print(f"\nDiagnostic subset:\n  {diagnostic_indices.size} deterministic fire/activity-balanced samples")
    print(f"\nFLARE baseline fire silhouette:\n  {_format_metric(baseline_metrics['fire_id_silhouette'], 6)}")
    print(f"\nFLARE final fire silhouette:\n  {_format_metric(final_metrics['fire_id_silhouette'], 6)}")
    print(f"\nFLARE baseline activity silhouette:\n  {_format_metric(baseline_metrics['activity_bin_silhouette'], 6)}")
    print(f"\nFLARE final activity silhouette:\n  {_format_metric(final_metrics['activity_bin_silhouette'], 6)}")
    print(f"\nOutput:\n  {output_dir / 'latent_alignment.pdf'}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, choices=ALLOWED_MODEL_SEEDS, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--samples-per-fire", type=int, default=150)
    parser.add_argument("--projection", choices=("umap", "pca"), default="umap")
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--diagnostic-samples", type=int, default=0, help="0 uses every selected sample.")
    parser.add_argument("--diagnostic-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete torch device")
    parser.add_argument("--annotate-metrics", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--population-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "all_data_seed_042/selected_samples.json",
        help="Optional existing complete-population target classification manifest.",
    )
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1:
        parser.error("batch size/CPU threads must be positive and workers nonnegative")
    if args.samples_per_fire < 1:
        parser.error("--samples-per-fire must be positive")
    if args.diagnostic_samples != 0 and args.diagnostic_samples < 100:
        parser.error("--diagnostic-samples must be 0 or at least 100")
    if args.n_neighbors < 2 or not 0.0 <= args.min_dist <= 1.0:
        parser.error("invalid UMAP parameters")
    torch.set_num_threads(int(args.cpu_threads))
    run(args)


if __name__ == "__main__":
    main()
