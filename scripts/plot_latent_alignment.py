#!/usr/bin/env python3
"""Create a paired, test-only FLARE latent-alignment figure and diagnostics.

The selection is frozen from target metadata before either checkpoint is
opened. Embeddings are global-average-pooled ``z_shared`` tensors returned by
the model's existing ``return_features=True`` interface. All diagnostics are
computed in that original pooled latent space, never in a 2-D projection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
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
    resolve_input_normalization_stats_path,
)


ALLOWED_MODEL_SEEDS = (42, 123, 2026)
TABLE2_SUMMARY = ROOT / "artifacts/table2_baselines/summary/table2_results.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/latent_alignment"
MODEL_ROWS = {"baseline": "cawfe_latte_baseline", "final": "cawfe_latte_final"}
PAPER_NAMES = {"baseline": "FLARE baseline", "final": "FLARE final", "ga": "GA (no MMD)"}
FIRE_DISPLAY_NAMES = {
    "CHIMNEYTOPS2": "Chimney Tops 2",
    "MCKINNEY": "McKinney",
    "RIM__0822__keepz_08": "Rim",
    "SPRINGS": "Springs",
    "THOMAS__1210__keepz_08": "Thomas",
    "TUOLUMNE_65": "Tuolumne 65",
    "WOOLSEY": "Woolsey",
}
ACTIVITY_DISPLAY_NAMES = {
    "no_fire": "No fire",
    "tiny_fire": "Tiny",
    "small_fire": "Small",
    "medium_fire": "Medium",
    "large_fire": "Large",
}
ACTIVITY_COLORS = {
    "no_fire": "#7A7A7A",
    "tiny_fire": "#56B4E9",
    "small_fire": "#009E73",
    "medium_fire": "#E69F00",
    "large_fire": "#D55E00",
}
METRIC_FIELDS = (
    "fire_id_silhouette",
    "activity_bin_silhouette",
    "mean_within_fire_distance",
    "mean_between_fire_distance",
    "between_within_fire_ratio",
    "nearest_neighbor_fire_id_agreement",
    "nearest_neighbor_activity_bin_agreement",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _resolved_config_path(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "resolved_config.yaml", run_dir / "configs/resolved_config.yaml"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _assert_complete_test_identity(run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    path = run_dir / "evaluation/test_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Locked held-out test metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("metric_scope") != "complete_locked_held_out_test"
        or payload.get("split") != "test"
        or payload.get("test_used_for_model_selection") is not False
        or int(payload.get("dataset_sample_count", -1)) <= 0
        or int(payload.get("evaluated_test_samples", -1)) != int(payload.get("dataset_sample_count", -2))
        or Path(str(payload.get("checkpoint", ""))).expanduser().resolve() != checkpoint.resolve()
    ):
        raise RuntimeError(f"Frozen test/checkpoint identity is incomplete or incompatible: {path}")
    return payload


def _resolve_table2_run(model_key: str, seed: int) -> dict[str, Any]:
    summary = json.loads(TABLE2_SUMMARY.read_text(encoding="utf-8"))
    row = next((item for item in summary.get("rows", []) if item.get("model_key") == MODEL_ROWS[model_key]), None)
    if row is None:
        raise FileNotFoundError(f"Table 2 summary has no {MODEL_ROWS[model_key]} row: {TABLE2_SUMMARY}")
    matches = [
        ROOT / directory
        for found_seed, directory in zip(row.get("seed_values", []), row.get("run_dirs", []))
        if int(found_seed) == int(seed)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen {model_key} run for seed={seed}, got {matches}")
    run_dir = matches[0].resolve()
    checkpoint = run_dir / "checkpoints/best_model.pt"
    config_path = _resolved_config_path(run_dir)
    if not checkpoint.is_file() or config_path is None:
        raise FileNotFoundError(f"Frozen run is incomplete: {run_dir}")
    config = load_config(config_path)
    if config.get("model", {}).get("architecture") != "cawfe_latte":
        raise RuntimeError(f"Frozen run is not a FLARE/CAWFE-Latte model: {config_path}")
    if int(config.get("training", {}).get("seed", config.get("seed", -1))) != int(seed):
        raise RuntimeError(f"Resolved model seed does not equal requested seed={seed}: {config_path}")
    finalist = config.get("final_training", {}).get("finalist")
    cawfe = config.get("cawfe_latte", {})
    if model_key == "baseline":
        if (
            finalist != "baseline"
            or cawfe.get("ablation", {}).get("name") != "baseline"
            or cawfe.get("post_fusion_backbone", {}).get("type") != "baseline_cnn"
            or bool(cawfe.get("fire_mmd", {}).get("enabled", False))
        ):
            raise RuntimeError(f"Left checkpoint is not the preserved FLARE baseline: {config_path}")
    else:
        if (
            finalist != "GA_Q2"
            or config.get("final_training", {}).get("source_ablation") != "GA_Q2_fire_domain_mmd"
            or cawfe.get("ablation", {}).get("name") != "GA_Q2_fire_domain_mmd"
            or not bool(cawfe.get("fire_mmd", {}).get("enabled", False))
            or cawfe.get("temporal_pooling", {}).get("type") == "attention"
        ):
            raise RuntimeError(f"FLARE final is not exact GA_Q2 without temporal-attention C: {config_path}")
    return {
        "key": model_key,
        "paper_name": PAPER_NAMES[model_key],
        "seed": int(seed),
        "run_dir": run_dir,
        "checkpoint": checkpoint.resolve(),
        "config_path": config_path,
        "config": config,
        "test_metrics": _assert_complete_test_identity(run_dir, checkpoint),
    }


def _ga_candidates(seed: int) -> list[Path]:
    roots = (
        ROOT / "artifacts/final_training/cawfe_latte/GA" / f"seed_{seed}",
        ROOT / "artifacts/final_training/cawfe_latte/GA_separate_decoder_resblocks" / f"seed_{seed}",
    )
    candidates: list[Path] = []
    for parent in roots:
        if not parent.is_dir():
            continue
        for child in [parent, *sorted((path for path in parent.iterdir() if path.is_dir()), reverse=True)]:
            if (child / "checkpoints/best_model.pt").is_file() and _resolved_config_path(child) is not None:
                candidates.append(child.resolve())
    return candidates


def _resolve_ga_run(seed: int, explicit: Path | None, final_run: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [explicit.expanduser().resolve()] if explicit is not None else _ga_candidates(seed)
    compatible: list[dict[str, Any]] = []
    final_cawfe = copy.deepcopy(dict(final_run["config"]["cawfe_latte"]))
    final_cawfe.pop("ablation", None)
    final_cawfe.pop("fire_mmd", None)
    for run_dir in candidates:
        checkpoint = run_dir / "checkpoints/best_model.pt"
        config_path = _resolved_config_path(run_dir)
        if not checkpoint.is_file() or config_path is None:
            continue
        config = load_config(config_path)
        cawfe = copy.deepcopy(dict(config.get("cawfe_latte", {})))
        ablation = cawfe.pop("ablation", {})
        mmd = cawfe.pop("fire_mmd", {})
        found_seed = int(config.get("training", {}).get("seed", config.get("seed", -1)))
        if (
            found_seed == int(seed)
            and ablation.get("name") == "GA_separate_decoder_resblocks"
            and not bool(mmd.get("enabled", False))
            and cawfe == final_cawfe
        ):
            compatible.append(
                {
                    "key": "ga",
                    "paper_name": PAPER_NAMES["ga"],
                    "seed": int(seed),
                    "run_dir": run_dir,
                    "checkpoint": checkpoint.resolve(),
                    "config_path": config_path,
                    "config": config,
                    "test_metrics": final_run["test_metrics"],
                }
            )
    if len(compatible) != 1:
        raise FileNotFoundError(
            "No unique compatible frozen full-training GA checkpoint exists for this seed. "
            "The repository contains the GA ablation config but no full-training GA checkpoint; "
            "this analysis will not retrain it automatically."
        )
    return compatible[0]


def _sample_index_path(config: Mapping[str, Any]) -> Path:
    dataloader = config.get("dataloader", {})
    configured = dataloader.get("sample_index_path")
    if configured not in (None, "", "null"):
        value = Path(str(configured)).expanduser()
        if not value.is_absolute():
            config_path = Path(str(config.get("config_path", config.get("_config_path", ROOT))))
            value = config_path.resolve().parent / value
        return value.resolve()
    dataset_root = Path(str(dataloader.get("dataset_root", config["processed_dataset"]["root"]))).expanduser().resolve()
    return (dataset_root / "indices/temporal" / f"samples_{dataloader['sample_pattern']}.jsonl").resolve()


def _protocol_signature(run: Mapping[str, Any]) -> dict[str, Any]:
    config = run["config"]
    dataloader = config["dataloader"]
    normalization = resolve_input_normalization_stats_path(config, must_exist=True)
    index = _sample_index_path(config)
    if not index.is_file():
        raise FileNotFoundError(index)
    return {
        "dataset_root": str(Path(str(dataloader.get("dataset_root", config["processed_dataset"]["root"]))).resolve()),
        "sample_pattern": str(dataloader["sample_pattern"]),
        "sample_index_path": str(index),
        "sample_index_sha256": sha256_file(index),
        "patch_size": int(dataloader["patch_size"]),
        "target_horizon": int(dataloader["target_horizon"]),
        "input_key": str(dataloader["input_key"]),
        "terrain_key": str(dataloader["terrain_key"]),
        "input_sequence_length": int(config["input_sequence_length"]),
        "prediction_horizon": int(config["prediction_horizon"]),
        "input_channels": int(config["model"]["input_channels"]),
        "normalization_path": str(normalization.resolve()),
        "normalization_sha256": sha256_file(normalization),
        "test_sample_count": int(run["test_metrics"]["dataset_sample_count"]),
    }


def _runtime_test_loader(config: Mapping[str, Any], batch_size: int, num_workers: int):
    runtime = copy.deepcopy(dict(config))
    runtime["return_metadata"] = True
    dataloader = dict(runtime.get("dataloader", {}))
    dataloader["return_metadata"] = True
    runtime["dataloader"] = dataloader
    data_loader = copy.deepcopy(dict(runtime.get("data_loader", {})))
    for split in ("train", "val", "test"):
        options = dict(data_loader.get(split, {}))
        options.update(
            {
                "num_workers": int(num_workers) if split == "test" else 0,
                "persistent_workers": bool(num_workers > 0) if split == "test" else False,
                "pin_memory": False,
            }
        )
        if split == "test":
            options["batch_size"] = int(batch_size)
            options["drop_last"] = False
        data_loader[split] = options
    data_loader["batch_size"] = int(batch_size)
    runtime["data_loader"] = data_loader
    train_loader, val_loader, test_loader = create_dataloaders(runtime)
    del train_loader, val_loader
    if test_loader is None or str(getattr(test_loader.dataset, "split", "")) != "test":
        raise RuntimeError("The resolved loader did not construct the held-out TEST split.")
    return runtime, test_loader


def _stable_group_seed(selection_seed: int, fire_name: str, activity_bin: str) -> int:
    payload = f"{selection_seed}|{fire_name}|{activity_bin}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def select_balanced_samples(
    classified: Sequence[Mapping[str, Any]], *, selection_seed: int, samples_per_fire: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select target-only samples by round-robin balancing bins within each fire."""

    if samples_per_fire < 1:
        raise ValueError("samples_per_fire must be positive.")
    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen_ids: set[str] = set()
    for item in classified:
        row = dict(item)
        sample_id = str(row["sample_id"])
        fire_name = str(row["fire_name"])
        activity_bin = str(row["activity_bin"])
        if sample_id in seen_ids:
            raise RuntimeError(f"Duplicate test sample ID: {sample_id}")
        if activity_bin not in ACTIVE_FRACTION_BIN_NAMES:
            raise ValueError(f"Noncanonical activity bin: {activity_bin}")
        seen_ids.add(sample_id)
        pools[fire_name][activity_bin].append(row)

    selected: list[dict[str, Any]] = []
    eligible_counts: dict[str, dict[str, int]] = {}
    selected_counts: dict[str, dict[str, int]] = {}
    for fire_name in sorted(pools):
        fire_pools: dict[str, list[dict[str, Any]]] = {}
        eligible_counts[fire_name] = {}
        for activity_bin in ACTIVE_FRACTION_BIN_NAMES:
            rows = sorted(pools[fire_name].get(activity_bin, []), key=lambda row: str(row["sample_id"]))
            random.Random(_stable_group_seed(selection_seed, fire_name, activity_bin)).shuffle(rows)
            fire_pools[activity_bin] = rows
            eligible_counts[fire_name][activity_bin] = len(rows)
        target_count = min(samples_per_fire, sum(len(rows) for rows in fire_pools.values()))
        cursors = {name: 0 for name in ACTIVE_FRACTION_BIN_NAMES}
        fire_selected: list[dict[str, Any]] = []
        while len(fire_selected) < target_count:
            progressed = False
            for activity_bin in ACTIVE_FRACTION_BIN_NAMES:
                cursor = cursors[activity_bin]
                rows = fire_pools[activity_bin]
                if cursor < len(rows) and len(fire_selected) < target_count:
                    fire_selected.append(rows[cursor])
                    cursors[activity_bin] += 1
                    progressed = True
            if not progressed:
                break
        if len(fire_selected) != target_count:
            raise RuntimeError(f"Balanced selection stalled for {fire_name}.")
        selected.extend(fire_selected)
        counts = Counter(str(row["activity_bin"]) for row in fire_selected)
        selected_counts[fire_name] = {name: int(counts.get(name, 0)) for name in ACTIVE_FRACTION_BIN_NAMES}
    if not selected or len({str(item["sample_id"]) for item in selected}) != len(selected):
        raise RuntimeError("Balanced selection is empty or contains duplicate sample IDs.")
    return selected, {
        "algorithm": "deterministic_per_fire_canonical_activity_bin_round_robin_v1",
        "selection_seed": int(selection_seed),
        "samples_per_fire_requested": int(samples_per_fire),
        "eligible_counts_by_fire_and_bin": eligible_counts,
        "selected_counts_by_fire_and_bin": selected_counts,
    }


def _selection_manifest(
    *, dataset: Any, signature: Mapping[str, Any], selection_seed: int, samples_per_fire: int
) -> dict[str, Any]:
    classified = classify_validation_records(dataset, progress_every=5000)
    if len(classified) != len(dataset):
        raise RuntimeError("Canonical target classifier did not cover the complete test dataset.")
    if any(dataset.records[int(item["index"])].get("split") != "test" for item in classified):
        raise RuntimeError("Selection pool contains a non-test record.")
    selected, audit = select_balanced_samples(
        classified, selection_seed=selection_seed, samples_per_fire=samples_per_fire
    )
    observed_test_fires = {str(record["fire_name"]) for record in dataset.records}
    records: list[dict[str, Any]] = []
    for rank, item in enumerate(selected):
        record = dataset.records[int(item["index"])]
        if str(record["sample_id"]) != str(item["sample_id"]) or record.get("split") != "test":
            raise RuntimeError("Selection index/sample identity changed during target-only selection.")
        records.append(
            {
                "selection_rank": rank,
                "dataset_index": int(item["index"]),
                "sample_id": str(item["sample_id"]),
                "split": "test",
                "fire_id": str(item["fire_name"]),
                "activity_bin": str(item["activity_bin"]),
                "active_fraction": float(item["active_fraction"]),
                "active_pixel_count": int(item["active_pixels"]),
                "patch": {key: int(record["patch"][key]) for key in ("y0", "x0", "height", "width")},
                "current_index": int(record["current_index"]),
                "target_index": int(record["target_index"]),
                "input_indices": [int(value) for value in record["input_indices"]],
                "target_path": str(record["target_path"]),
            }
        )
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "split": "test",
        "selection_uses_predictions_or_features": False,
        "selection_completed_before_checkpoint_loading": True,
        "selection_basis": "ground_truth_target_mask_only",
        "fire_mask_threshold": FIRE_MASK_THRESHOLD,
        "canonical_activity_bin_names": list(ACTIVE_FRACTION_BIN_NAMES),
        "canonical_activity_bin_boundaries": list(ACTIVE_FRACTION_BOUNDARIES),
        "dataset_sample_count": len(dataset),
        "held_out_test_fires": sorted(observed_test_fires),
        "sample_index_path": signature["sample_index_path"],
        "sample_index_sha256": signature["sample_index_sha256"],
        "selected_sample_count": len(records),
        "audit": audit,
        "samples": records,
    }


def _validate_manual_test_fires(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected = set(str(value) for value in config.get("manual_fire_split", {}).get("test_fires", []))
    observed = set(str(value) for value in manifest["held_out_test_fires"])
    if not expected or observed != expected:
        raise RuntimeError(f"Held-out test fire identity mismatch: expected={sorted(expected)} observed={sorted(observed)}")
    if any(sample.get("split") != "test" or sample.get("fire_id") not in expected for sample in manifest["samples"]):
        raise RuntimeError("Selected samples are not exclusively from held-out TEST fires.")


def _load_model(run: Mapping[str, Any]) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    config = run["config"]
    model = build_model_from_config(config, input_channels=int(config["model"]["input_channels"])).to("cpu")
    checkpoint = load_checkpoint(run["checkpoint"], map_location="cpu")
    validate_checkpoint_model_compatibility(model, checkpoint, run["checkpoint"])
    load_model_state_dict_compatible(model, checkpoint, run["checkpoint"])
    model.eval()
    return model, checkpoint


def _collated_sample_ids(metadata: Any) -> list[str]:
    if not isinstance(metadata, Mapping) or "sample_id" not in metadata:
        raise RuntimeError("Selected inference batch lacks sample_id metadata.")
    value = metadata["sample_id"]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if torch.is_tensor(value):
        return [str(item) for item in value.tolist()]
    return [str(value)]


def extract_pooled_latents(
    *,
    left_run: Mapping[str, Any],
    final_run: Mapping[str, Any],
    test_loader: Any,
    selection: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run both frozen models on each exact same batch and pool only z_shared."""

    dataset = test_loader.dataset
    selected = list(selection["samples"])
    indices = [int(item["dataset_index"]) for item in selected]
    selected_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        persistent_workers=bool(num_workers > 0),
        pin_memory=False,
    )
    normalizer = build_input_normalizer_for_loader(
        test_loader,
        torch.device("cpu"),
        int(left_run["config"]["model"]["input_channels"]),
        left_run["config"],
    )
    left_model, left_checkpoint = _load_model(left_run)
    final_model, final_checkpoint = _load_model(final_run)
    left_features: list[np.ndarray] = []
    final_features: list[np.ndarray] = []
    position = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(selected_loader):
            x, y, extra = unpack_batch(batch)
            expected = selected[position : position + int(x.shape[0])]
            actual_ids = _collated_sample_ids(extra.get("metadata"))
            expected_ids = [str(item["sample_id"]) for item in expected]
            if actual_ids != expected_ids:
                raise RuntimeError(f"Selected inference order changed: expected={expected_ids} actual={actual_ids}")
            classified = classify_fire_masks(y[:, 2])
            fractions = [float(value) for value in classified["active_fraction"].tolist()]
            bins = [active_fraction_bin_name(value) for value in fractions]
            if bins != [str(item["activity_bin"]) for item in expected]:
                raise RuntimeError("Canonical activity labels changed between selection and inference.")
            x = apply_input_normalization(x.float(), normalizer, left_run["config"])
            terrain = extra.get("terrain")
            terrain = terrain.float() if torch.is_tensor(terrain) else None
            pooled: list[np.ndarray] = []
            for model in (left_model, final_model):
                output = model(x, return_features=True) if terrain is None else model(x, terrain=terrain, return_features=True)
                if not isinstance(output, Mapping) or "z_shared" not in output:
                    raise RuntimeError("FLARE return_features=True did not return canonical z_shared.")
                latent = output["z_shared"]
                if not torch.is_tensor(latent) or latent.ndim != 4 or int(latent.shape[0]) != int(x.shape[0]):
                    raise RuntimeError(f"Invalid z_shared shape: {getattr(latent, 'shape', None)}")
                if not torch.isfinite(latent).all():
                    raise RuntimeError("z_shared contains NaN or Inf.")
                pooled.append(latent.mean(dim=(-2, -1)).float().cpu().numpy())
            left_features.append(pooled[0])
            final_features.append(pooled[1])
            position += int(x.shape[0])
            if (batch_index + 1) % 10 == 0 or position == len(selected):
                print(f"LATENT EXTRACTION: {position}/{len(selected)} test samples", flush=True)
    del left_model, final_model, selected_loader
    left_array = np.concatenate(left_features, axis=0).astype(np.float32, copy=False)
    final_array = np.concatenate(final_features, axis=0).astype(np.float32, copy=False)
    if left_array.shape[0] != len(selected) or final_array.shape[0] != len(selected):
        raise RuntimeError("Latent extraction did not return one vector per selected sample.")
    if left_array.shape[1] != final_array.shape[1]:
        raise RuntimeError(f"Shared-latent dimensions differ: {left_array.shape} vs {final_array.shape}")
    return left_array, final_array, {
        "left_checkpoint_epoch": left_checkpoint.get("epoch"),
        "final_checkpoint_epoch": final_checkpoint.get("epoch"),
        "latent_key": "z_shared",
        "latent_tensor_layout": "B,D,H,W",
        "pooling": "unweighted_global_average_over_H_W",
        "pooled_dimension": int(left_array.shape[1]),
        "map_location": "cpu",
        "model_mode": "eval",
        "autograd": False,
        "same_tensor_batches_supplied_to_both_models": True,
    }


def latent_diagnostics(features: np.ndarray, fire_ids: Sequence[str], activity_bins: Sequence[str]) -> dict[str, Any]:
    """Compute Euclidean diagnostics in original pooled latent space."""

    values = np.asarray(features, dtype=np.float64)
    fire = np.asarray(fire_ids, dtype="U")
    activity = np.asarray(activity_bins, dtype="U")
    if values.ndim != 2 or values.shape[0] != fire.shape[0] or values.shape[0] != activity.shape[0]:
        raise ValueError("Latent features and labels are not aligned.")
    if values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("Latent diagnostics require at least three finite feature vectors.")
    distances = pairwise_distances(values, metric="euclidean")
    np.fill_diagonal(distances, 0.0)

    def silhouette(labels: np.ndarray) -> float | None:
        counts = Counter(labels.tolist())
        if len(counts) < 2 or any(count < 2 for count in counts.values()):
            return None
        return float(silhouette_score(distances, labels, metric="precomputed"))

    upper = np.triu_indices(values.shape[0], k=1)
    pair_values = distances[upper]
    same_fire = fire[upper[0]] == fire[upper[1]]
    within = float(pair_values[same_fire].mean()) if bool(same_fire.any()) else None
    between = float(pair_values[~same_fire].mean()) if bool((~same_fire).any()) else None
    ratio = None if within in (None, 0.0) or between is None else float(between / within)
    nearest = distances.copy()
    np.fill_diagonal(nearest, np.inf)
    nearest_index = nearest.argmin(axis=1)
    return {
        "space": "original_high_dimensional_global_average_pooled_z_shared",
        "distance": "euclidean",
        "sample_count": int(values.shape[0]),
        "latent_dimension": int(values.shape[1]),
        "fire_id_silhouette": silhouette(fire),
        "activity_bin_silhouette": silhouette(activity),
        "mean_within_fire_distance": within,
        "mean_between_fire_distance": between,
        "between_within_fire_ratio": ratio,
        "nearest_neighbor_fire_id_agreement": float(np.mean(fire[nearest_index] == fire)),
        "nearest_neighbor_activity_bin_agreement": float(np.mean(activity[nearest_index] == activity)),
        "fire_counts": dict(sorted(Counter(fire.tolist()).items())),
        "activity_bin_counts": {name: int(Counter(activity.tolist()).get(name, 0)) for name in ACTIVE_FRACTION_BIN_NAMES},
    }


def _pca_projection(features: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    reducer = PCA(n_components=2, svd_solver="full", random_state=int(seed))
    coordinates = reducer.fit_transform(np.asarray(features, dtype=np.float64))
    return coordinates, {"method": "pca", "explained_variance_ratio": [float(value) for value in reducer.explained_variance_ratio_]}


def project_separately(
    left_features: np.ndarray,
    final_features: np.ndarray,
    *,
    requested: str,
    projection_seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fallback_reason = None
    if requested == "umap":
        try:
            import umap  # type: ignore[import-not-found]
        except ImportError as exc:
            fallback_reason = f"UMAP unavailable ({exc}); used separate PCA fits instead."
            warnings.warn(fallback_reason, RuntimeWarning, stacklevel=2)
        else:
            neighbors = min(max(2, int(n_neighbors)), int(left_features.shape[0]) - 1)

            def fit(values: np.ndarray) -> np.ndarray:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=neighbors,
                    min_dist=float(min_dist),
                    metric="euclidean",
                    random_state=int(projection_seed),
                    transform_seed=int(projection_seed),
                    n_jobs=1,
                )
                return np.asarray(reducer.fit_transform(values), dtype=np.float64)

            return fit(left_features), fit(final_features), {
                "requested": "umap",
                "effective": "umap",
                "fit_separately_per_model": True,
                "projection_seed": int(projection_seed),
                "n_neighbors": neighbors,
                "min_dist": float(min_dist),
                "metric": "euclidean",
                "fallback_reason": None,
            }
    left_coordinates, left_pca = _pca_projection(left_features, projection_seed)
    final_coordinates, final_pca = _pca_projection(final_features, projection_seed)
    return left_coordinates, final_coordinates, {
        "requested": requested,
        "effective": "pca",
        "fit_separately_per_model": True,
        "projection_seed": int(projection_seed),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "fallback_reason": fallback_reason,
        "left": left_pca,
        "final": final_pca,
    }


def _clean_fire_name(name: str) -> str:
    if name in FIRE_DISPLAY_NAMES:
        return FIRE_DISPLAY_NAMES[name]
    return name.split("__", 1)[0].replace("_", " ").strip().title()


def _fire_colors(fires: Sequence[str]) -> dict[str, Any]:
    ordered = sorted(set(str(value) for value in fires))
    if len(ordered) <= 10:
        colors = list(plt.get_cmap("tab10").colors)
    else:
        colors = [
            *plt.get_cmap("tab20").colors,
            *plt.get_cmap("tab20b").colors,
            *plt.get_cmap("tab20c").colors,
        ]
    return {name: colors[index % len(colors)] for index, name in enumerate(ordered)}


def _format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "N/A"


def draw_latent_figure(
    *,
    output_dir: Path,
    left_coordinates: np.ndarray,
    final_coordinates: np.ndarray,
    fire_ids: Sequence[str],
    activity_bins: Sequence[str],
    left_metrics: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    left_title: str,
    annotate_metrics: bool,
    fire_display_names: Mapping[str, str] | None = None,
    large_dataset_style: bool = False,
) -> dict[str, str]:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 6.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fire = np.asarray(fire_ids, dtype="U")
    activity = np.asarray(activity_bins, dtype="U")
    fire_colors = _fire_colors(fire_ids)
    many_fires = len(fire_colors) > 12
    use_large_style = bool(large_dataset_style or fire.shape[0] > 10000)
    figure_height = 7.4 if many_fires else 5.7
    fig, axes = plt.subplots(2, 2, figsize=(7.16, figure_height), facecolor="white")
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.93 if many_fires else 0.91,
        bottom=0.11 if many_fires else 0.13,
        wspace=0.12,
        hspace=1.05 if many_fires else 0.42,
    )
    coordinates = (left_coordinates, final_coordinates)
    model_metrics = (left_metrics, final_metrics)
    marker_size = 0.65 if use_large_style else 8
    marker_alpha = 0.16 if use_large_style else 0.58
    for column, title in enumerate((left_title, PAPER_NAMES["final"])):
        axes[0, column].set_title(title, fontweight="semibold", pad=7)
        for fire_name in sorted(fire_colors):
            mask = fire == fire_name
            axes[0, column].scatter(coordinates[column][mask, 0], coordinates[column][mask, 1], s=marker_size, alpha=marker_alpha, color=fire_colors[fire_name], edgecolors="none", rasterized=use_large_style)
        for activity_bin in ACTIVE_FRACTION_BIN_NAMES:
            mask = activity == activity_bin
            if bool(mask.any()):
                axes[1, column].scatter(coordinates[column][mask, 0], coordinates[column][mask, 1], s=marker_size, alpha=marker_alpha, color=ACTIVITY_COLORS[activity_bin], edgecolors="none", rasterized=use_large_style)
        if annotate_metrics:
            axes[0, column].text(0.02, 0.98, f"Fire silhouette = {_format_metric(model_metrics[column]['fire_id_silhouette'])}", transform=axes[0, column].transAxes, ha="left", va="top", fontsize=6.8, bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.9})
            axes[1, column].text(0.02, 0.98, f"Activity silhouette = {_format_metric(model_metrics[column]['activity_bin_silhouette'])}", transform=axes[1, column].transAxes, ha="left", va="top", fontsize=6.8, bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.9})
    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        axis.text(-0.02, 1.02, f"({label})", transform=axis.transAxes, ha="right", va="bottom", fontweight="semibold")
        axis.set_xlabel("Projection 1")
        axis.set_ylabel("Projection 2")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("0.55")
        axis.spines["bottom"].set_color("0.55")
    axes[0, 0].text(-0.18, 0.5, "Colored by fire ID", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontweight="semibold")
    axes[1, 0].text(-0.18, 0.5, "Colored by activity regime", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center", fontweight="semibold")
    display_names = dict(fire_display_names or {})
    fire_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=3.5 if many_fires else 4, markerfacecolor=fire_colors[name], markeredgewidth=0, label=display_names.get(name, _clean_fire_name(name))) for name in sorted(fire_colors)]
    activity_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4, markerfacecolor=ACTIVITY_COLORS[name], markeredgewidth=0, label=ACTIVITY_DISPLAY_NAMES[name]) for name in ACTIVE_FRACTION_BIN_NAMES if bool((activity == name).any())]
    fig.legend(handles=fire_handles, loc="center", bbox_to_anchor=(0.54, 0.515 if many_fires else 0.505), ncol=6 if many_fires else 4, frameon=False, columnspacing=0.8 if many_fires else 1.2, handletextpad=0.25 if many_fires else 0.35, fontsize=5.2 if many_fires else 6.8)
    fig.legend(handles=activity_handles, loc="lower center", bbox_to_anchor=(0.54, 0.006 if many_fires else 0.012), ncol=5, frameon=False, columnspacing=1.4, handletextpad=0.35)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"latent_alignment.{name}" for name in ("pdf", "svg", "png")}
    vector_dpi = 600 if use_large_style else "figure"
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white", dpi=vector_dpi, metadata={"Title": "FLARE latent alignment"})
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white", dpi=vector_dpi)
    fig.savefig(paths["png"], bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)
    return {key: str(path.resolve()) for key, path in paths.items()}


def _training_mmd_metadata(final_run: Mapping[str, Any]) -> dict[str, Any]:
    mmd_config = dict(final_run["config"].get("cawfe_latte", {}).get("fire_mmd", {}))
    history_path = final_run["run_dir"] / "history/epoch_history.csv"
    if not history_path.is_file():
        history_path = final_run["run_dir"] / "training_history.csv"
    by_epoch: list[dict[str, Any]] = []
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_loss = row.get("train_mmd_loss")
                raw_valid = row.get("mmd_valid_batch_fraction", row.get("train_mmd_valid_batch_fraction"))
                try:
                    loss = float(raw_loss) if raw_loss not in (None, "") else None
                    valid = float(raw_valid) if raw_valid not in (None, "") else None
                except ValueError:
                    continue
                if loss is not None and math.isfinite(loss):
                    by_epoch.append({"epoch": int(float(row["epoch"])), "train_mmd_loss": loss, "mmd_valid_batch_fraction": valid, "is_best_epoch": str(row.get("is_best_epoch", row.get("is_best", ""))).lower() in {"1", "true"}})
    losses = [float(row["train_mmd_loss"]) for row in by_epoch]
    best = next((row for row in by_epoch if row["is_best_epoch"]), None)
    return {
        "config": mmd_config,
        "history_path": str(history_path.resolve()) if history_path.is_file() else None,
        "by_epoch": by_epoch,
        "summary": {
            "epoch_count": len(by_epoch),
            "mean_train_mmd_loss": float(np.mean(losses)) if losses else None,
            "minimum_train_mmd_loss": min(losses) if losses else None,
            "final_train_mmd_loss": losses[-1] if losses else None,
            "best_checkpoint_epoch_train_mmd_loss": None if best is None else best["train_mmd_loss"],
        },
    }


def _write_coordinate_csv(path: Path, selection: Mapping[str, Any], left_coordinates: np.ndarray, final_coordinates: np.ndarray, left_key: str) -> None:
    fields = ["sample_id", "dataset_index", "fire_id", "fire_display_name", "activity_bin", "activity_display_name", "active_fraction", f"{left_key}_x", f"{left_key}_y", "final_x", "final_y"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(selection["samples"]):
            writer.writerow({"sample_id": sample["sample_id"], "dataset_index": sample["dataset_index"], "fire_id": sample["fire_id"], "fire_display_name": _clean_fire_name(str(sample["fire_id"])), "activity_bin": sample["activity_bin"], "activity_display_name": ACTIVITY_DISPLAY_NAMES[str(sample["activity_bin"])], "active_fraction": repr(float(sample["active_fraction"])), f"{left_key}_x": repr(float(left_coordinates[index, 0])), f"{left_key}_y": repr(float(left_coordinates[index, 1])), "final_x": repr(float(final_coordinates[index, 0])), "final_y": repr(float(final_coordinates[index, 1]))})
    temporary.replace(path)


def _write_metrics_csv(path: Path, metrics: Mapping[str, Any], names: Mapping[str, str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fields = ["model_key", "model_name", *METRIC_FIELDS, "sample_count", "latent_dimension"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_key, item in metrics.items():
            writer.writerow({"model_key": model_key, "model_name": names[model_key], **{field: item.get(field) for field in METRIC_FIELDS}, "sample_count": item["sample_count"], "latent_dimension": item["latent_dimension"]})
    temporary.replace(path)


def _save_pooled_latents(path: Path, selection: Mapping[str, Any], left_features: np.ndarray, final_features: np.ndarray, left_key: str) -> None:
    samples = selection["samples"]
    arrays: dict[str, Any] = {
        "sample_ids": np.asarray([item["sample_id"] for item in samples], dtype="U"),
        "fire_ids": np.asarray([item["fire_id"] for item in samples], dtype="U"),
        "activity_bins": np.asarray([item["activity_bin"] for item in samples], dtype="U"),
        "active_fractions": np.asarray([item["active_fraction"] for item in samples], dtype=np.float64),
        "dataset_indices": np.asarray([item["dataset_index"] for item in samples], dtype=np.int64),
        "final_features": np.asarray(final_features, dtype=np.float32),
        "left_model_key": np.asarray(left_key),
        f"{left_key}_features": np.asarray(left_features, dtype=np.float32),
    }
    _atomic_npz(path, **arrays)


def _load_pooled_latents(path: Path, selection: Mapping[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        left_key = str(archive["left_model_key"])
        sample_ids = [str(value) for value in archive["sample_ids"].tolist()]
        expected = [str(item["sample_id"]) for item in selection["samples"]]
        if sample_ids != expected:
            raise RuntimeError("Cached pooled latents do not match selected sample IDs/order.")
        return left_key, np.asarray(archive[f"{left_key}_features"]), np.asarray(archive["final_features"])


def _comparison_metrics(left: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, Any]:
    return {f"final_minus_left_{field}": None if left.get(field) is None or final.get(field) is None else float(final[field]) - float(left[field]) for field in METRIC_FIELDS}


def _render_and_save(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    selection: Mapping[str, Any],
    left_key: str,
    left_features: np.ndarray,
    final_features: np.ndarray,
    run_metadata: Mapping[str, Any],
    extraction_metadata: Mapping[str, Any],
    mmd_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    samples = selection["samples"]
    fire_ids = [str(item["fire_id"]) for item in samples]
    activity_bins = [str(item["activity_bin"]) for item in samples]
    left_metrics = latent_diagnostics(left_features, fire_ids, activity_bins)
    final_metrics = latent_diagnostics(final_features, fire_ids, activity_bins)
    projection_seed = int(args.umap_seed if args.umap_seed is not None else args.selection_seed)
    left_coordinates, final_coordinates, projection_metadata = project_separately(left_features, final_features, requested=args.projection, projection_seed=projection_seed, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    figure_paths = draw_latent_figure(output_dir=output_dir, left_coordinates=left_coordinates, final_coordinates=final_coordinates, fire_ids=fire_ids, activity_bins=activity_bins, left_metrics=left_metrics, final_metrics=final_metrics, left_title=PAPER_NAMES[left_key], annotate_metrics=bool(args.annotate_metrics))
    _write_coordinate_csv(output_dir / "embedding_coordinates.csv", selection, left_coordinates, final_coordinates, left_key)
    metrics = {left_key: left_metrics, "final": final_metrics}
    names = {left_key: PAPER_NAMES[left_key], "final": PAPER_NAMES["final"]}
    metrics_payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "diagnostic_space": "original_high_dimensional_global_average_pooled_z_shared",
        "projection_not_used_for_diagnostics": True,
        "models": metrics,
        "comparison": _comparison_metrics(left_metrics, final_metrics),
        "mmd_training_metadata": mmd_metadata,
        "interpretation_rule": {"fire_specific_alignment": "lower fire-ID silhouette/agreement/ratio suggests less fire-specific clustering", "forecast_relevant_structure": "higher activity-bin silhouette/agreement suggests better activity-regime structure", "conclusion_is_not_forced": True},
    }
    _atomic_json(output_dir / "latent_metrics.json", metrics_payload)
    _write_metrics_csv(output_dir / "latent_metrics.csv", metrics, names)
    caveat = "The GA versus GA-Q2 control shares architecture and isolates the configured MMD term." if left_key == "ga" else "FLARE baseline and FLARE final differ in post-fusion backbone depth/type and decoder structure as well as MMD; this comparison cannot attribute latent changes solely to MMD."
    figure_metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "terminology": {"CAWFE-Latte": "FLARE", "GA_Q2": "FLARE final", "baseline": "FLARE baseline"},
        "model_seed": int(args.model_seed),
        "selection_seed": int(args.selection_seed),
        "projection_seed": projection_seed,
        "control": args.control,
        "left_model_key": left_key,
        "models": run_metadata,
        "latent_feature": extraction_metadata,
        "selection_path": str((output_dir / "selected_samples.json").resolve()),
        "selection_sha256": sha256_file(output_dir / "selected_samples.json"),
        "selected_sample_count": len(samples),
        "samples_per_fire": dict(sorted(Counter(fire_ids).items())),
        "activity_bin_counts": {name: int(Counter(activity_bins).get(name, 0)) for name in ACTIVE_FRACTION_BIN_NAMES},
        "canonical_activity_bins": list(ACTIVE_FRACTION_BIN_NAMES),
        "projection": projection_metadata,
        "projection_coordinate_systems_independently_fit_and_not_directly_comparable": True,
        "diagnostics_computed_before_projection_in_original_space": True,
        "candidate_ranking": "none",
        "candidate_selection_policy": "inspect robustness and choose a clear representative; never rank by FLARE advantage",
        "ga_control_available_in_repository": bool(_ga_candidates(args.model_seed)),
        "attribution_caveat": caveat,
        "annotate_metrics": bool(args.annotate_metrics),
        "outputs": {**figure_paths, "selected_samples": str((output_dir / "selected_samples.json").resolve()), "pooled_latents": str((output_dir / "pooled_latents.npz").resolve()), "embedding_coordinates": str((output_dir / "embedding_coordinates.csv").resolve()), "latent_metrics_json": str((output_dir / "latent_metrics.json").resolve()), "latent_metrics_csv": str((output_dir / "latent_metrics.csv").resolve())},
        "git_commit": _git_commit(),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__},
    }
    _atomic_json(output_dir / "figure_metadata.json", figure_metadata)
    return {"metrics": metrics_payload, "metadata": figure_metadata}


def _run_record(run: Mapping[str, Any]) -> dict[str, Any]:
    return {"key": run["key"], "paper_name": run["paper_name"], "seed": run["seed"], "run_dir": str(run["run_dir"]), "checkpoint": str(run["checkpoint"]), "checkpoint_sha256": sha256_file(run["checkpoint"]), "resolved_config": str(run["config_path"]), "resolved_config_sha256": sha256_file(run["config_path"]), "finalist": run["config"].get("final_training", {}).get("finalist"), "source_ablation": run["config"].get("final_training", {}).get("source_ablation")}


def _print_report(*, args: argparse.Namespace, output_dir: Path, left_run: Mapping[str, Any] | None, final_run: Mapping[str, Any] | None, selection: Mapping[str, Any], result: Mapping[str, Any] | None) -> None:
    print("\nLATENT ALIGNMENT ANALYSIS")
    if left_run is not None:
        print(f"\n{left_run['paper_name']} checkpoint:\n  {left_run['checkpoint']}")
    if final_run is not None:
        print(f"\nFLARE final checkpoint:\n  {final_run['checkpoint']}")
    print(f"\nModel seed:\n  {args.model_seed}")
    print(f"\nSelection seed:\n  {args.selection_seed}")
    print(f"\nTest samples used:\n  {selection['selected_sample_count']}")
    print("\nSamples per fire:")
    for fire_name, count in sorted(Counter(str(item["fire_id"]) for item in selection["samples"]).items()):
        print(f"  {_clean_fire_name(fire_name)}: {count}")
    print("\nActivity bins:")
    activity_counts = Counter(str(item["activity_bin"]) for item in selection["samples"])
    for name in ACTIVE_FRACTION_BIN_NAMES:
        print(f"  {ACTIVITY_DISPLAY_NAMES[name]}: {activity_counts.get(name, 0)}")
    print("\nLatent feature:\n  Z_shared -> global average pooled")
    if result is not None:
        projection = result["metadata"]["projection"]
        print(f"\nProjection:\n  {projection['effective'].upper()}")
        if projection.get("fallback_reason"):
            print(f"  FALLBACK: {projection['fallback_reason']}")
        metrics = result["metrics"]["models"]
        left_key = result["metadata"]["left_model_key"]
        print(f"\n{PAPER_NAMES[left_key]} fire silhouette:\n  {_format_metric(metrics[left_key]['fire_id_silhouette'], 6)}")
        print(f"\nFLARE final fire silhouette:\n  {_format_metric(metrics['final']['fire_id_silhouette'], 6)}")
        print(f"\n{PAPER_NAMES[left_key]} activity silhouette:\n  {_format_metric(metrics[left_key]['activity_bin_silhouette'], 6)}")
        print(f"\nFLARE final activity silhouette:\n  {_format_metric(metrics['final']['activity_bin_silhouette'], 6)}")
        print(f"\nOutput:\n  {(output_dir / 'latent_alignment.pdf').resolve()}")
    else:
        print(f"\nSelection output:\n  {(output_dir / 'selected_samples.json').resolve()}")
    print("\nGenerate 10 candidate figures:\n  python scripts/generate_latent_alignment_candidates.py --num-candidates 10 " f"--model-seed {args.model_seed} --projection {args.projection}")


def _existing_output_guard(output_dir: Path, overwrite: bool) -> None:
    completed = [output_dir / name for name in ("latent_alignment.pdf", "latent_alignment.svg", "latent_alignment.png", "pooled_latents.npz") if (output_dir / name).exists()]
    if completed and not overwrite:
        raise FileExistsError(f"Completed latent outputs already exist in {output_dir}: {completed}. Use --render-only to redraw cached features or --overwrite to replace outputs.")


def run(args: argparse.Namespace) -> Path:
    output_dir = (args.output_dir or (args.output_root / f"seed_{int(args.selection_seed):03d}")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selected_samples.json"
    pooled_path = output_dir / "pooled_latents.npz"
    if args.render_only:
        if not selection_path.is_file() or not pooled_path.is_file() or not (output_dir / "figure_metadata.json").is_file():
            raise FileNotFoundError("--render-only requires selected_samples.json, pooled_latents.npz, and figure_metadata.json.")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        prior_metadata = json.loads((output_dir / "figure_metadata.json").read_text(encoding="utf-8"))
        if int(prior_metadata["model_seed"]) != int(args.model_seed):
            raise RuntimeError("--render-only model seed differs from cached features.")
        left_key, left_features, final_features = _load_pooled_latents(pooled_path, selection)
        args.control = "ga" if left_key == "ga" else "baseline"
        prior_metrics_path = output_dir / "latent_metrics.json"
        prior_metrics = json.loads(prior_metrics_path.read_text(encoding="utf-8")) if prior_metrics_path.is_file() else {}
        result = _render_and_save(args=args, output_dir=output_dir, selection=selection, left_key=left_key, left_features=left_features, final_features=final_features, run_metadata=prior_metadata.get("models", {}), extraction_metadata=prior_metadata.get("latent_feature", {}), mmd_metadata=prior_metrics.get("mmd_training_metadata", {}))
        _print_report(args=args, output_dir=output_dir, left_run=None, final_run=None, selection=selection, result=result)
        return output_dir

    _existing_output_guard(output_dir, bool(args.overwrite))
    final_run = _resolve_table2_run("final", args.model_seed)
    left_run = _resolve_table2_run("baseline", args.model_seed) if args.control == "baseline" else _resolve_ga_run(args.model_seed, args.ga_run_dir, final_run)
    left_signature = _protocol_signature(left_run)
    final_signature = _protocol_signature(final_run)
    if left_signature != final_signature:
        raise RuntimeError(f"Left and FLARE-final models do not share exact dataset/preprocessing/normalization/test identity.\nleft={left_signature}\nfinal={final_signature}")
    runtime_config, test_loader = _runtime_test_loader(left_run["config"], args.batch_size, args.num_workers)
    if len(test_loader.dataset) != int(left_signature["test_sample_count"]):
        raise RuntimeError("Constructed TEST loader count differs from the locked Table 2 test count.")
    if selection_path.is_file() and not args.reselect:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if int(selection.get("audit", {}).get("selection_seed", -1)) != int(args.selection_seed) or int(selection.get("audit", {}).get("samples_per_fire_requested", -1)) != int(args.samples_per_fire) or selection.get("sample_index_sha256") != left_signature["sample_index_sha256"]:
            raise RuntimeError(f"Existing selection is incompatible; use a new output directory or --reselect: {selection_path}")
    else:
        if args.reselect and pooled_path.exists():
            raise RuntimeError("Cannot reselect after pooled model features exist; choose a new output directory.")
        selection = _selection_manifest(dataset=test_loader.dataset, signature=left_signature, selection_seed=args.selection_seed, samples_per_fire=args.samples_per_fire)
        _validate_manual_test_fires(runtime_config, selection)
        _atomic_json(selection_path, selection)
    _validate_manual_test_fires(runtime_config, selection)
    if not selection_path.is_file():
        raise RuntimeError("Selection was not made durable before feature extraction.")
    if args.selection_only:
        _print_report(args=args, output_dir=output_dir, left_run=left_run, final_run=final_run, selection=selection, result=None)
        return output_dir
    left_features, final_features, extraction_metadata = extract_pooled_latents(left_run=left_run, final_run=final_run, test_loader=test_loader, selection=selection, batch_size=args.batch_size, num_workers=args.num_workers)
    _save_pooled_latents(pooled_path, selection, left_features, final_features, str(left_run["key"]))
    run_metadata = {str(left_run["key"]): _run_record(left_run), "final": _run_record(final_run)}
    result = _render_and_save(args=args, output_dir=output_dir, selection=selection, left_key=str(left_run["key"]), left_features=left_features, final_features=final_features, run_metadata=run_metadata, extraction_metadata=extraction_metadata, mmd_metadata=_training_mmd_metadata(final_run))
    _print_report(args=args, output_dir=output_dir, left_run=left_run, final_run=final_run, selection=selection, result=result)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, choices=ALLOWED_MODEL_SEEDS, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--projection", choices=("umap", "pca"), default="umap")
    parser.add_argument("--umap-seed", type=int, help="Defaults to --selection-seed.")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--samples-per-fire", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--annotate-metrics", action="store_true")
    parser.add_argument("--control", choices=("baseline", "ga"), default="baseline")
    parser.add_argument("--ga-run-dir", type=Path, help="Optional explicit frozen full-training GA run; never trained automatically.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-only", action="store_true", help="Freeze/audit the target-only selection without loading checkpoints.")
    parser.add_argument("--render-only", action="store_true", help="Redraw from pooled_latents.npz without running inference.")
    parser.add_argument("--reselect", action="store_true", help="Replace a selection only when no pooled features exist.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace completed outputs in the selected directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2")
    if not 0.0 <= args.min_dist <= 1.0:
        parser.error("--min-dist must be in [0,1]")
    if args.samples_per_fire < 1:
        parser.error("--samples-per-fire must be positive")
    if args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1:
        parser.error("batch size/CPU threads must be positive and workers nonnegative")
    if args.control == "baseline" and args.ga_run_dir is not None:
        parser.error("--ga-run-dir is only valid with --control ga")
    torch.set_num_threads(int(args.cpu_threads))
    run(args)


if __name__ == "__main__":
    main()
