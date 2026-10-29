#!/usr/bin/env python3
"""Plot the complete FLARE architecture-ablation trade-offs.

Only explicitly named full-validation fields are accepted.  The script never
falls back to training, screening-subset, or held-out-test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "ablation_tradeoffs.pdf"
EXPECTED_CONFIGURATION_COUNT = 33

# Centralized paper-style controls.  Values are deliberately conservative for
# an IEEE/TGRS two-column figure and can be adjusted without touching plotting
# logic.
STYLE: dict[str, Any] = {
    "figure_width": 7.16,
    "figure_height": 2.78,
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "base_font_size": 8.25,
    "axis_label_size": 8.5,
    "title_size": 8.8,
    "tick_size": 7.5,
    "annotation_size": 7.2,
    "legend_size": 7.4,
    "ordinary_marker_size": 18.0,
    "finalist_marker_size": 43.0,
    "ordinary_alpha": 0.72,
    "grid_alpha": 0.22,
    "grid_line_width": 0.45,
    "spine_line_width": 0.55,
    "ordinary_edge_width": 0.25,
    "finalist_edge_width": 0.75,
    "final_edge_width": 1.25,
    "leader_line_width": 0.45,
    "pareto_line_width": 0.8,
    "pareto_alpha": 0.50,
    "ordinary_color": "#B8BCC2",
    "ordinary_edge_color": "#8A8F96",
    "pareto_color": "#6F7782",
    "better_color": "#7E858E",
    "colors": {
        "Base": "#4B4F56",
        "CGP": "#2474B7",
        "GA-Q2": "#D45532",
        "GPK": "#2E8B57",
    },
    "grayscale_colors": {
        "Base": "#777777",
        "CGP": "#A6A6A6",
        "GA-Q2": "#181818",
        "GPK": "#4B4B4B",
    },
    "grayscale_line_styles": {
        "Base": "-",
        "CGP": "--",
        "GA-Q2": "-.",
        "GPK": ":",
    },
    "annotation_offsets": {
        0: {"Base": (-18, 10), "CGP": (-6, 2), "GA-Q2": (-8, 3), "GPK": (-22, 13)},
        1: {"Base": (-18, 11), "CGP": (-14, -13), "GA-Q2": (-20, 8), "GPK": (-20, 8)},
        2: {"Base": (-18, 10), "CGP": (-18, 10), "GA-Q2": (-20, -13), "GPK": (-20, 10)},
    },
    "better_arrow_positions": {
        0: {"text": (0.48, 0.80), "tip": (0.62, 0.66)},
        1: {"text": (0.53, 0.80), "tip": (0.67, 0.66)},
        2: {"text": (0.42, 0.20), "tip": (0.56, 0.07)},
    },
    "legend_location": "upper center",
    "legend_bbox": (0.5, 0.985),
    "legend_columns": 5,
    "x_padding_fraction": 0.055,
    "y_padding_fraction": 0.075,
    "subplot_left": 0.080,
    "subplot_right": 0.995,
    "subplot_bottom": 0.205,
    "subplot_top": 0.765,
    "subplot_wspace": 0.36,
}


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "dice": ("Full Val Dice", "full_val_dice"),
    "hrr_log_mae": (
        "Full Val HRR Log MAE",
        "Full Val Energy Log MAE",
        "full_val_hrr_log_mae",
        "full_val_energy_log_mae",
    ),
    "surface_mae": ("Full Val Surface MAE", "full_val_surface_mae"),
    "active_canopy_mae": (
        "Full Val Active Canopy MAE",
        "full_val_active_canopy_mae",
    ),
    "full_val_count": (
        "Full Val Total Patches",
        "full_val_total_patch_count",
    ),
}

NAME_FIELDS = ("Ablation", "ablation", "Configuration", "configuration", "name")

HIGHLIGHT_ALIASES: dict[str, tuple[str, ...]] = {
    "Base": ("baseline", "base", "cawfe_latte_baseline", "flare_baseline"),
    "CGP": ("cgp", "cgp_separate_decoder_temporal_attention_mamba"),
    "GA-Q2": ("ga_q2", "ga_q2_fire_domain_mmd"),
    "GPK": ("gpk", "gpk_mamba_no_terrain"),
}

PANELS = (
    ("hrr_log_mae", "(a) HRR forecast", "HRR Log MAE"),
    ("surface_mae", "(b) Surface consumption", "Surface MAE"),
    ("active_canopy_mae", "(c) Active canopy", "Active Canopy MAE"),
)

# These are guards against accidentally loading the legacy 400-patch summary,
# never a substitute for the machine-readable data.
EXPECTED_BEST = {
    "dice": ("CGP", 0.8848, 0.0030, "max"),
    "hrr_log_mae": ("GPK", 0.0402, 0.0040, "min"),
    "surface_mae": ("GPK", 0.0377, 0.0050, "min"),
    "active_canopy_mae": ("CGP", 0.9099, 0.0300, "min"),
}


def _normalize_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _display_name(value: str) -> str | None:
    normalized = _normalize_name(value)
    for display, aliases in HIGHLIGHT_ALIASES.items():
        if normalized in {_normalize_name(alias) for alias in aliases}:
            return display
    return None


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _load_raw_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = next(
                (payload[key] for key in ("rows", "results", "ablations") if isinstance(payload.get(key), list)),
                None,
            )
            if rows is None:
                raise ValueError("JSON does not contain a rows/results/ablations list")
        else:
            raise ValueError("JSON root must be a list or object")
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("JSON result rows must be objects")
        return list(rows)
    raise ValueError(f"Unsupported result format: {path.suffix}; use CSV or JSON")


def _coerce_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows = _load_raw_rows(path)
    rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows, start=2):
        name_value = _first_present(raw, NAME_FIELDS)
        if name_value in (None, ""):
            raise ValueError(f"row {row_index} has no configuration name")
        record: dict[str, Any] = {"name": str(name_value).strip()}
        for metric in ("dice", "hrr_log_mae", "surface_mae", "active_canopy_mae"):
            value = _first_present(raw, FIELD_ALIASES[metric])
            if value in (None, ""):
                aliases = ", ".join(FIELD_ALIASES[metric])
                raise ValueError(f"row {row_index} lacks explicit full-validation field: {aliases}")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {row_index} has non-numeric {metric}: {value!r}") from exc
            if not math.isfinite(number):
                raise ValueError(f"row {row_index} has non-finite {metric}: {number}")
            record[metric] = number

        count_value = _first_present(raw, FIELD_ALIASES["full_val_count"])
        record["full_val_count"] = None if count_value in (None, "") else float(count_value)
        rows.append(record)
    return rows


def _candidate_result_paths() -> Iterable[Path]:
    canonical_dir = REPO_ROOT / "artifacts" / "ablations" / "cawfe_latte"
    yield canonical_dir / "ablation_results.csv"
    yield canonical_dir / "ablation_results.json"

    # Home-directory snapshots are read-only and preserve the canonical results
    # even if bulky run directories were later cleaned from the working tree.
    snapshot_root = Path.home() / ".snapshot"
    snapshot_pattern = (
        f"*/{REPO_ROOT.name}/artifacts/ablations/cawfe_latte/ablation_results.csv"
    )
    if snapshot_root.is_dir():
        yield from sorted(snapshot_root.glob(snapshot_pattern), reverse=True)


def _validate_rows(rows: Sequence[Mapping[str, Any]], source: Path) -> dict[str, Mapping[str, Any]]:
    names = [str(row["name"]) for row in rows]
    if len(names) != EXPECTED_CONFIGURATION_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CONFIGURATION_COUNT} rows, found {len(names)}"
        )
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate configuration rows: {duplicates}")

    full_counts = [row.get("full_val_count") for row in rows]
    present_counts = [float(value) for value in full_counts if value is not None]
    if present_counts:
        if not all(math.isfinite(value) and value > 400 for value in present_counts):
            raise ValueError("full-validation counts include a non-finite or <=400 value")
        if len(present_counts) != len(rows):
            raise ValueError("full-validation patch counts are present for only some rows")
        if len({int(value) for value in present_counts}) != 1:
            raise ValueError("configurations were not evaluated on the same full-validation set")

    highlighted: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        display = _display_name(str(row["name"]))
        if display is not None:
            if display in highlighted:
                raise ValueError(f"multiple rows match highlighted alias {display}")
            highlighted[display] = row
    missing = sorted(set(HIGHLIGHT_ALIASES) - set(highlighted))
    if missing:
        raise ValueError(f"missing highlighted configuration(s): {', '.join(missing)}")

    _validate_expected_best(rows)
    return highlighted


def _best_row(rows: Sequence[Mapping[str, Any]], metric: str, direction: str) -> Mapping[str, Any]:
    key = lambda row: float(row[metric])
    return (max if direction == "max" else min)(rows, key=key)


def _validate_expected_best(rows: Sequence[Mapping[str, Any]]) -> None:
    for metric, (expected_name, expected_value, tolerance, direction) in EXPECTED_BEST.items():
        best = _best_row(rows, metric, direction)
        actual_name = _display_name(str(best["name"])) or str(best["name"])
        actual_value = float(best[metric])
        if actual_name != expected_name or abs(actual_value - expected_value) > tolerance:
            raise ValueError(
                f"{metric} safety check failed: got {actual_name}={actual_value:.6f}; "
                f"expected approximately {expected_name}={expected_value:.4f}. "
                "The source may contain screening or otherwise incompatible metrics."
            )


def load_results(explicit_path: Path | None) -> tuple[Path, list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    if explicit_path is not None:
        candidates = [explicit_path.expanduser().resolve()]
    else:
        candidates = list(dict.fromkeys(path.resolve() for path in _candidate_result_paths()))

    errors: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            rows = _coerce_rows(path)
            highlighted = _validate_rows(rows, path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            if explicit_path is not None:
                break
            continue
        return path, rows, highlighted

    detail = "\n".join(f"  - {message}" for message in errors) or "  - no candidate file exists"
    raise RuntimeError(
        "Could not find a compatible complete 33-run full-validation ablation summary.\n"
        f"Checked:\n{detail}\n"
        "Pass the canonical CSV explicitly with --results."
    )


def _padded_limits(values: Sequence[float], fraction: float) -> tuple[float, float]:
    lower, upper = min(values), max(values)
    span = upper - lower
    padding = (span if span > 0 else max(abs(lower), 1.0) * 0.05) * fraction
    return lower - padding, upper + padding


def _pareto_front(rows: Sequence[Mapping[str, Any]], error_metric: str) -> list[Mapping[str, Any]]:
    front: list[Mapping[str, Any]] = []
    for candidate in rows:
        x_value = float(candidate["dice"])
        y_value = float(candidate[error_metric])
        dominated = any(
            float(other["dice"]) >= x_value
            and float(other[error_metric]) <= y_value
            and (
                float(other["dice"]) > x_value
                or float(other[error_metric]) < y_value
            )
            for other in rows
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: float(row["dice"]))


def _legend_handles(theme: str) -> list[Line2D]:
    colors = STYLE["colors"] if theme == "color" else STYLE["grayscale_colors"]
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor=STYLE["ordinary_color"],
            markeredgecolor=STYLE["ordinary_edge_color"],
            markeredgewidth=STYLE["ordinary_edge_width"],
            alpha=STYLE["ordinary_alpha"],
            label="All ablations",
        )
    ]
    for display in ("Base", "CGP", "GA-Q2", "GPK"):
        handles.append(
            Line2D(
                [],
                [],
                color=colors[display],
                linestyle=(
                    "none"
                    if theme == "color"
                    else STYLE["grayscale_line_styles"][display]
                ),
                linewidth=0.8,
                marker="o",
                markersize=5.5,
                markerfacecolor=colors[display],
                markeredgecolor="#202226",
                markeredgewidth=(
                    STYLE["final_edge_width"]
                    if display == "GA-Q2"
                    else STYLE["finalist_edge_width"]
                ),
                label=display,
            )
        )
    return handles


def plot_tradeoffs(
    rows: Sequence[Mapping[str, Any]],
    highlighted: Mapping[str, Mapping[str, Any]],
    output: Path,
    *,
    show_pareto: bool,
    annotate_finalists: bool,
    theme: str,
    better_arrow: bool,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": STYLE["font_family"],
            "font.serif": STYLE["font_serif"],
            "font.size": STYLE["base_font_size"],
            "axes.labelsize": STYLE["axis_label_size"],
            "axes.titlesize": STYLE["title_size"],
            "xtick.labelsize": STYLE["tick_size"],
            "ytick.labelsize": STYLE["tick_size"],
            "legend.fontsize": STYLE["legend_size"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(STYLE["figure_width"], STYLE["figure_height"]),
        facecolor="white",
    )
    figure.subplots_adjust(
        left=STYLE["subplot_left"],
        right=STYLE["subplot_right"],
        bottom=STYLE["subplot_bottom"],
        top=STYLE["subplot_top"],
        wspace=STYLE["subplot_wspace"],
    )

    x_values = [float(row["dice"]) for row in rows]
    shared_x_limits = _padded_limits(x_values, STYLE["x_padding_fraction"])
    colors = STYLE["colors"] if theme == "color" else STYLE["grayscale_colors"]

    for panel_index, (axis, (metric, title, ylabel)) in enumerate(zip(axes, PANELS)):
        y_values = [float(row[metric]) for row in rows]
        axis.set_axisbelow(True)
        axis.grid(
            True,
            color="#B9BEC5",
            alpha=STYLE["grid_alpha"],
            linewidth=STYLE["grid_line_width"],
        )
        axis.scatter(
            x_values,
            y_values,
            s=STYLE["ordinary_marker_size"],
            marker="o",
            c=STYLE["ordinary_color"],
            edgecolors=STYLE["ordinary_edge_color"],
            linewidths=STYLE["ordinary_edge_width"],
            alpha=STYLE["ordinary_alpha"],
            zorder=2,
        )

        if show_pareto:
            front = _pareto_front(rows, metric)
            axis.plot(
                [float(row["dice"]) for row in front],
                [float(row[metric]) for row in front],
                color=STYLE["pareto_color"],
                linewidth=STYLE["pareto_line_width"],
                alpha=STYLE["pareto_alpha"],
                linestyle="--",
                zorder=1,
            )

        for display in ("Base", "CGP", "GA-Q2", "GPK"):
            row = highlighted[display]
            x_value, y_value = float(row["dice"]), float(row[metric])
            edge_width = (
                STYLE["final_edge_width"]
                if display == "GA-Q2"
                else STYLE["finalist_edge_width"]
            )
            axis.scatter(
                [x_value],
                [y_value],
                s=STYLE["finalist_marker_size"],
                marker="o",
                c=[colors[display]],
                edgecolors="#202226",
                linewidths=edge_width,
                alpha=0.98,
                zorder=4,
            )
            if annotate_finalists:
                offset = STYLE["annotation_offsets"][panel_index][display]
                line_style = (
                    "-"
                    if theme == "color"
                    else STYLE["grayscale_line_styles"][display]
                )
                axis.annotate(
                    display,
                    xy=(x_value, y_value),
                    xytext=offset,
                    textcoords="offset points",
                    ha="right" if offset[0] < 0 else "left",
                    va="center",
                    fontsize=STYLE["annotation_size"],
                    fontweight="bold" if display == "GA-Q2" else "normal",
                    color="#202226",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": colors[display],
                        "linewidth": STYLE["leader_line_width"],
                        "linestyle": line_style,
                        "shrinkA": 1.5,
                        "shrinkB": 3.0,
                    },
                    annotation_clip=True,
                    zorder=5,
                )

        if better_arrow:
            arrow_position = STYLE["better_arrow_positions"][panel_index]
            axis.annotate(
                "better",
                xy=arrow_position["tip"],
                xytext=arrow_position["text"],
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=6.2,
                fontstyle="italic",
                color=STYLE["better_color"],
                alpha=0.72,
                ha="center",
                va="center",
                arrowprops={
                    "arrowstyle": "->",
                    "color": STYLE["better_color"],
                    "linewidth": 0.55,
                    "mutation_scale": 7,
                },
                zorder=1,
            )

        axis.set_xlim(*shared_x_limits)
        axis.set_ylim(*_padded_limits(y_values, STYLE["y_padding_fraction"]))
        axis.set_xlabel("Fire-mask Dice", labelpad=3.0)
        axis.set_ylabel(ylabel, labelpad=3.0)
        axis.set_title(title, pad=5.0)
        axis.tick_params(axis="both", width=STYLE["spine_line_width"], length=2.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(STYLE["spine_line_width"])
        axis.spines["bottom"].set_linewidth(STYLE["spine_line_width"])

    figure.legend(
        handles=_legend_handles(theme),
        loc=STYLE["legend_location"],
        bbox_to_anchor=STYLE["legend_bbox"],
        ncol=STYLE["legend_columns"],
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    output = output.expanduser().resolve()
    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must end in .pdf; the sibling .png is generated automatically")
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_suffix(".png")
    figure.savefig(output, format="pdf", facecolor="white")
    figure.savefig(preview, format="png", dpi=600, facecolor="white")
    plt.close(figure)
    return preview


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _print_diagnostics(
    source: Path,
    rows: Sequence[Mapping[str, Any]],
    highlighted: Mapping[str, Mapping[str, Any]],
    output: Path,
    preview: Path,
) -> None:
    print("Detected configurations (33):")
    for name in sorted(str(row["name"]) for row in rows):
        print(f"  {name}")

    print("\nHighlighted full-validation values:")
    for display in ("Base", "CGP", "GA-Q2", "GPK"):
        row = highlighted[display]
        print(
            f"  {display}: Dice={float(row['dice']):.6f}, "
            f"HRR Log MAE={float(row['hrr_log_mae']):.6f}, "
            f"Surface MAE={float(row['surface_mae']):.6f}, "
            f"Active Canopy MAE={float(row['active_canopy_mae']):.6f}"
        )

    best_specs = (
        ("Best Dice", "dice", "max"),
        ("Lowest HRR Log MAE", "hrr_log_mae", "min"),
        ("Lowest Surface MAE", "surface_mae", "min"),
        ("Lowest Active Canopy MAE", "active_canopy_mae", "min"),
    )
    print("\nIndependent metric optima (not a weighted ranking):")
    for label, metric, direction in best_specs:
        row = _best_row(rows, metric, direction)
        display = _display_name(str(row["name"])) or str(row["name"])
        print(f"  {label}: {display} ({float(row[metric]):.6f})")

    print("\nABLATION TRADE-OFF FIGURE\n")
    print("Source results:")
    print(f"  {_relative_or_absolute(source)}\n")
    print("Configurations plotted:")
    print(f"  {len(rows)}\n")
    print("Metrics:")
    print("  Dice vs HRR Log MAE")
    print("  Dice vs Surface MAE")
    print("  Dice vs Active Canopy MAE\n")
    print("Highlighted:")
    print("  Base")
    print("  CGP")
    print("  GA-Q2")
    print("  GPK\n")
    print("Full-validation metrics used:")
    print("  YES\n")
    print("Screening metrics used:")
    print("  NO\n")
    print("Weighted score used:")
    print("  NO\n")
    print("Output:")
    print(f"  {_relative_or_absolute(output)}")
    print(f"  {_relative_or_absolute(preview)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full-validation trade-offs for the complete 33-run FLARE ablation."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Canonical ablation_results.csv/json. By default, search active artifacts "
            "and then read-only home snapshots."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show-pareto", action="store_true", help="Show pairwise Pareto fronts.")
    parser.add_argument(
        "--annotate-finalists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate Base, CGP, GA-Q2, and GPK (default: on).",
    )
    parser.add_argument("--theme", choices=("color", "grayscale"), default="color")
    parser.add_argument(
        "--better-arrow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the restrained lower-right 'better' arrow (default: on).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source, rows, highlighted = load_results(args.results)
    output = args.output.expanduser().resolve()
    preview = plot_tradeoffs(
        rows,
        highlighted,
        output,
        show_pareto=args.show_pareto,
        annotate_finalists=args.annotate_finalists,
        theme=args.theme,
        better_arrow=args.better_arrow,
    )
    _print_diagnostics(source, rows, highlighted, output, preview)


if __name__ == "__main__":
    main()
