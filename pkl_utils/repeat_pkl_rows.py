#!/usr/bin/env python3
"""Concatenate all rows of a pkl file to itself n times and save the result."""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Load a pkl file and concatenate its rows to itself n times."
    )
    parser.add_argument("pkl_path", type=str, help="Path to the .pkl file")
    parser.add_argument(
        "n",
        type=int,
        help="Number of times to repeat the rows (result will have n × original rows)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path (default: same dir as input, name <original>_x<n>.pkl e.g. data_x300.pkl)",
    )
    args = parser.parse_args()

    path = Path(args.pkl_path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    if args.n < 1:
        print("n must be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        obj = pd.read_pickle(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(obj, pd.DataFrame):
        result = pd.concat([obj] * args.n, ignore_index=True)
        out_path = Path(args.output) if args.output else path.parent / f"{path.stem}_x{args.n}.pkl"
        result.to_pickle(out_path)
        print(f"Saved {len(result)} rows to {out_path}")
    else:
        try:
            rows = list(obj)
        except TypeError:
            print("pkl content is not a DataFrame or row sequence", file=sys.stderr)
            sys.exit(1)
        repeated = rows * args.n
        out_path = Path(args.output) if args.output else path.parent / f"{path.stem}_x{args.n}.pkl"
        pd.to_pickle(repeated, out_path)
        print(f"Saved {len(repeated)} rows to {out_path}")


if __name__ == "__main__":
    main()
