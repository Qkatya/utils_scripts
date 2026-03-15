#!/usr/bin/env python3
"""Load a pkl file and print its summary: columns, row count, and head."""
import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Load a pkl file and print its summary.")
    parser.add_argument("path", type=str, help="Path to the .pkl file")
    args = parser.parse_args()

    path = args.path
    try:
        obj = pd.read_pickle(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(obj, pd.DataFrame):
        df = obj
        print("Columns:", list(df.columns))
        print("Number of rows:", len(df))
        print("\nHead:")
        with pd.option_context("display.max_columns", None):
            print(df.head())
    else:
        print(f"Type: {type(obj).__name__}")
        if hasattr(obj, "__len__"):
            print("Length:", len(obj))
        print("\nHead (repr of first 5 items):")
        try:
            for i, item in enumerate(obj):
                if i >= 5:
                    break
                print(f"  [{i}] {repr(item)[:200]}")
        except TypeError:
            print(repr(obj)[:500])


if __name__ == "__main__":
    main()
