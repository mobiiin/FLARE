#!/usr/bin/env python3
"""Generate unranked FLARE latent-alignment candidates for seed robustness.

Candidate ``i`` uses selection seed ``i`` and projection seed ``i``.  The
wrapper intentionally records no score or ordering beyond that seed number;
the candidates are for robustness inspection, not cherry-picking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = ROOT / "scripts/plot_latent_alignment.py"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/latent_alignment"
ALLOWED_MODEL_SEEDS = (42, 123, 2026)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--model-seed", type=int, choices=ALLOWED_MODEL_SEEDS, default=42)
    parser.add_argument("--projection", choices=("umap", "pca"), default="umap")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--samples-per-fire", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--annotate-metrics", action="store_true")
    parser.add_argument("--control", choices=("baseline", "ga"), default="baseline")
    parser.add_argument("--ga-run-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing inference or writing candidate outputs.",
    )
    return parser


def candidate_command(args: argparse.Namespace, seed: int) -> list[str]:
    output_dir = args.output_root.expanduser().resolve() / f"seed_{seed:03d}"
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--model-seed",
        str(args.model_seed),
        "--selection-seed",
        str(seed),
        "--umap-seed",
        str(seed),
        "--projection",
        args.projection,
        "--n-neighbors",
        str(args.n_neighbors),
        "--min-dist",
        str(args.min_dist),
        "--samples-per-fire",
        str(args.samples_per_fire),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--cpu-threads",
        str(args.cpu_threads),
        "--control",
        args.control,
        "--output-dir",
        str(output_dir),
    ]
    if args.ga_run_dir is not None:
        command.extend(("--ga-run-dir", str(args.ga_run_dir.expanduser().resolve())))
    if args.annotate_metrics:
        command.append("--annotate-metrics")
    if args.overwrite:
        command.append("--overwrite")
    if args.selection_only:
        command.append("--selection-only")
    return command


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_candidates < 1:
        parser.error("--num-candidates must be positive")
    if args.control == "baseline" and args.ga_run_dir is not None:
        parser.error("--ga-run-dir is only valid with --control ga")
    print(
        "Generating unranked latent-alignment candidates. "
        "No candidate will be scored or ranked by FLARE advantage.",
        flush=True,
    )
    for seed in range(args.num_candidates):
        command = candidate_command(args, seed)
        print(f"\nCANDIDATE seed_{seed:03d}\n  " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    print(
        f"\nCompleted {args.num_candidates} unranked candidate(s) under "
        f"{args.output_root.expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
