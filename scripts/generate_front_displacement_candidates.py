#!/usr/bin/env python3
"""Generate GT/input-only front-displacement selection sheets for several seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.plot_fire_front_displacement import DEFAULT_CONFIG, DEFAULT_OUTPUT, main as plot_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--selection-only', action='store_true',
                        help='Accepted for symmetry; candidate generation is always GT/input-only.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for seed in dict.fromkeys(args.selection_seeds):
        print(f'Generating GT/input-only selection for seed {seed}', flush=True)
        plot_main(['--selection-seed', str(seed), '--selection-only',
                   '--config', str(args.config), '--output-dir', str(args.output_dir)])


if __name__ == '__main__':
    main()
