"""
Filter a pkl by removing rows whose feature .npy file does not exist.

Path per row: features_base_path / run_path / {tar_id}.npy
Prints summary: number of deleted rows and their full paths.

Uses one directory listing per unique run_path (not per row), so speed stays
consistent and is much faster on large splits.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


FEATURES_BASE_PATH = Path(
    "/mnt/A3000/Recordings/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features"
)


def npy_path_for_row(run_path: str, tar_id: str, base: Path) -> Path:
    """Build the expected .npy path for a row (run_path may have leading/trailing slashes)."""
    run = run_path.strip("/")
    return base / run / f"{tar_id}.npy"


def main():
    parser = argparse.ArgumentParser(
        description="Remove pkl rows whose feature .npy file is missing."
    )
    parser.add_argument("pkl", type=Path, help="Input pickle file path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output pkl path (default: overwrite input)",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=FEATURES_BASE_PATH,
        help="Features base path (default: Q_Features v2_200fps_energy_std_sobel_stcorr)",
    )
    args = parser.parse_args()

    pkl_path = args.pkl.resolve()
    if not pkl_path.is_file():
        raise SystemExit(f"Input file not found: {pkl_path}")

    out_path = (args.output or pkl_path).resolve()
    base_path = args.base_path

    print(f"Loading {pkl_path} ...")
    df = pd.read_pickle(pkl_path)
    if "run_path" not in df.columns or "tar_id" not in df.columns:
        raise SystemExit("DataFrame must have columns 'run_path' and 'tar_id'.")

    df["tar_id"] = df["tar_id"].astype(str)
    n_before = len(df)

    # Group by run_path: one directory listing per dir instead of one exists() per row
    df["_run"] = df["run_path"].str.strip("/")
    groups = df.groupby("_run", sort=False)

    indices_to_drop = []
    missing_paths = []
    for run_key, grp in tqdm(groups, desc="Checking .npy paths", unit="dir"):
        dir_path = base_path / run_key
        if not dir_path.is_dir():
            for idx, row in grp.iterrows():
                indices_to_drop.append(idx)
                missing_paths.append(str(npy_path_for_row(row["run_path"], row["tar_id"], base_path)))
            continue
        existing = {f.name for f in dir_path.iterdir() if f.suffix == ".npy"}
        for idx, row in grp.iterrows():
            if f"{row['tar_id']}.npy" not in existing:
                indices_to_drop.append(idx)
                missing_paths.append(str(npy_path_for_row(row["run_path"], row["tar_id"], base_path)))

    df_filtered = df.drop(columns=["_run"]).drop(index=indices_to_drop).copy()
    n_deleted = len(indices_to_drop)

    print(f"\nRows before: {n_before}")
    print(f"Rows deleted (missing .npy): {n_deleted}")
    print(f"Rows after: {len(df_filtered)}")

    if n_deleted > 0:
        print(f"\nDeleted row paths ({n_deleted}):")
        for p in missing_paths:
            print(p)
    else:
        print("\nNo rows deleted.")

    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.to_pickle(out_path)
    print(f"\nSaved filtered DataFrame to {out_path}")


if __name__ == "__main__":
    main()
