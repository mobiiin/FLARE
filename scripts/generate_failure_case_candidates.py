#!/usr/bin/env python3
"""Generate seeds 0..N-1 from an objectively ranked FLARE final failure pool."""
from __future__ import annotations

import argparse

from scripts.plot_failure_case import CRITERIA, analyze_failure_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--criterion', choices=CRITERIA, default='composite')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--num-candidates', type=int, default=10)
    parser.add_argument('--model-seed', type=int, choices=(42, 123, 2026), default=42)
    parser.add_argument('--allow-no-fire', action='store_true')
    parser.add_argument('--show-error-map', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--layout', choices=('compact', 'wide'), default='compact')
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.num_candidates < 1:
        parser.error('--top-k and --num-candidates must be positive')
    output, selections = analyze_failure_cases(criterion=args.criterion, top_k=args.top_k,
        selection_seeds=range(args.num_candidates), model_seed=args.model_seed,
        allow_no_fire=args.allow_no_fire, show_error_map=args.show_error_map,
        layout=args.layout, write_index=True)
    print(f'GENERATED CANDIDATES: {len(selections)}')
    print(f'OUTPUT DIRECTORY: {output}')
    return output


if __name__ == '__main__':
    main()
