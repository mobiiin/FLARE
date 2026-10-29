#!/usr/bin/env python3
"""Generate seeded, ground-truth-selected background/weak-fire paper figures."""
from __future__ import annotations

import argparse

from scripts.plot_background_weak_fire import generate_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-candidates', type=int, default=10)
    parser.add_argument('--model-seed', type=int, choices=(42, 123, 2026), default=42)
    args = parser.parse_args(argv)
    if args.num_candidates < 1:
        parser.error('--num-candidates must be positive')
    output = generate_candidates(range(args.num_candidates), args.model_seed)
    print(f'OUTPUT DIRECTORY: {output}')
    return output


if __name__ == '__main__':
    main()
