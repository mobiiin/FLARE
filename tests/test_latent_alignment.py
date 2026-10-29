from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from scripts.plot_latent_alignment import (
    ACTIVE_FRACTION_BIN_NAMES,
    draw_latent_figure,
    latent_diagnostics,
    project_separately,
    select_balanced_samples,
)


def _classified_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for fire_name in ("FIRE_A", "FIRE_B"):
        for activity_bin in ACTIVE_FRACTION_BIN_NAMES:
            for repetition in range(6):
                rows.append(
                    {
                        "index": index,
                        "sample_id": f"{fire_name}-{activity_bin}-{repetition}",
                        "fire_name": fire_name,
                        "activity_bin": activity_bin,
                        "active_fraction": 0.0,
                        "active_pixels": 0,
                    }
                )
                index += 1
    return rows


def test_balanced_selection_is_deterministic_and_bin_balanced() -> None:
    rows = _classified_records()
    first, audit = select_balanced_samples(rows, selection_seed=7, samples_per_fire=15)
    repeated, _ = select_balanced_samples(rows, selection_seed=7, samples_per_fire=15)
    changed, _ = select_balanced_samples(rows, selection_seed=8, samples_per_fire=15)

    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in repeated]
    assert [row["sample_id"] for row in first] != [row["sample_id"] for row in changed]
    assert len(first) == 30
    assert audit["algorithm"].endswith("round_robin_v1")
    for fire_name in ("FIRE_A", "FIRE_B"):
        counts = Counter(row["activity_bin"] for row in first if row["fire_name"] == fire_name)
        assert counts == Counter({name: 3 for name in ACTIVE_FRACTION_BIN_NAMES})


def test_diagnostics_use_high_dimensional_features() -> None:
    rng = np.random.default_rng(4)
    fire = np.repeat(["A", "B", "C"], 10)
    activity = np.tile(np.repeat(ACTIVE_FRACTION_BIN_NAMES[:3], [4, 3, 3]), 3)
    features = rng.normal(size=(30, 8))
    features[:, 0] += np.repeat([0.0, 2.0, 4.0], 10)
    diagnostics = latent_diagnostics(features, fire, activity)

    assert diagnostics["space"].endswith("z_shared")
    assert diagnostics["latent_dimension"] == 8
    assert diagnostics["sample_count"] == 30
    assert diagnostics["mean_between_fire_distance"] > diagnostics["mean_within_fire_distance"]
    assert 0.0 <= diagnostics["nearest_neighbor_fire_id_agreement"] <= 1.0


def test_pca_is_fit_separately_and_figure_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    left = rng.normal(size=(30, 7))
    final = rng.normal(size=(30, 7)) * 3.0 + 8.0
    left_coordinates, final_coordinates, metadata = project_separately(
        left,
        final,
        requested="pca",
        projection_seed=3,
        n_neighbors=15,
        min_dist=0.1,
    )
    fire = np.repeat(["CHIMNEYTOPS2", "MCKINNEY", "WOOLSEY"], 10)
    activity = np.tile(np.repeat(ACTIVE_FRACTION_BIN_NAMES[:3], [4, 3, 3]), 3)
    left_metrics = latent_diagnostics(left, fire, activity)
    final_metrics = latent_diagnostics(final, fire, activity)
    paths = draw_latent_figure(
        output_dir=tmp_path,
        left_coordinates=left_coordinates,
        final_coordinates=final_coordinates,
        fire_ids=fire,
        activity_bins=activity,
        left_metrics=left_metrics,
        final_metrics=final_metrics,
        left_title="FLARE baseline",
        annotate_metrics=True,
    )

    assert left_coordinates.shape == final_coordinates.shape == (30, 2)
    assert metadata["effective"] == "pca"
    assert metadata["fit_separately_per_model"] is True
    for path in paths.values():
        assert Path(path).stat().st_size > 0
