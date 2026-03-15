#!/usr/bin/env python3
"""Filter expressive blendshapes pkl and split into train/test by person.

Filters out:
- All rows belonging to people who have glasses (has_glasses True on any of their rows).
- Rows where frame_num <= 20 (intro/outro screens).
- Rows whose feature .npy file does not exist (path: features_base_path / run_path / {tar_id}.npy).

Splits by person (run_path prefix before the last '/'): 20% of people -> test, 80% -> train.
Saves to same directory with _train.pkl and _test.pkl suffix.
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


FEATURES_BASE_PATH = Path(
    "/mnt/A3000/Recordings/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features"
)


def npy_path_for_row(run_path: str, tar_id: str, base: Path) -> Path:
    """Build the expected .npy path for a row (run_path may have leading/trailing slashes)."""
    run = run_path.strip("/") if isinstance(run_path, str) else ""
    return base / run / f"{tar_id}.npy"


def get_person_id(run_path: str) -> str:
    """Person = run_path prefix before the last '/' (e.g. 2026/02/18/PretzelTrill-142026)."""
    if pd.isna(run_path):
        return ""
    return run_path.rsplit("/", 1)[0] if "/" in run_path else run_path


def main():
    parser = argparse.ArgumentParser(
        description="Filter pkl (exclude people with glasses, low frame_num) and split train/test by person."
    )
    parser.add_argument("path", type=str, help="Path to the .pkl file")
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of people for test set (default 0.2)",
    )
    parser.add_argument(
        "--min-frame-num",
        type=int,
        default=20,
        help="Drop rows with frame_num <= this (default 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test person split",
    )
    parser.add_argument(
        "--output-stem",
        "-o",
        type=str,
        default="expressive_blendshapes_no_glasses_110326",
        help="Output file base name without _train/_test (default: expressive_blendshapes_no_glasses_110326)",
    )
    parser.add_argument(
        "--features-base-path",
        type=Path,
        default=FEATURES_BASE_PATH,
        help="Features base path for .npy existence check (default: Q_Features v2_200fps_energy_std_sobel_stcorr)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_pickle(path)
    n_orig = len(df)

    # Person = run_path prefix before last '/'
    df = df.copy()
    df["_person_id"] = df["run_path"].map(get_person_id)

    # Filter 1: drop all people who have glasses (any row with has_glasses True)
    persons_with_glasses = set(
        df.loc[df["has_glasses"] == True, "_person_id"].dropna().unique()
    )
    persons_with_glasses.discard("")
    df = df[~df["_person_id"].isin(persons_with_glasses)]
    n_after_glasses = len(df)
    print(f"Dropped {n_orig - n_after_glasses} rows (all people with glasses)")

    # Filter 2: low frame_num (intro/outro screens)
    mask_low_frames = df["frame_num"] <= args.min_frame_num
    df = df[~mask_low_frames]
    n_after_frames = len(df)
    print(f"Dropped {n_after_glasses - n_after_frames} rows (frame_num <= {args.min_frame_num})")

    # Filter 3: drop rows whose feature .npy file does not exist (one dir listing per run_path)
    if "tar_id" not in df.columns:
        raise ValueError("DataFrame must have column 'tar_id' for .npy existence check.")
    df["tar_id"] = df["tar_id"].astype(str)
    df["_run"] = df["run_path"].astype(str).str.strip("/")
    base_path = args.features_base_path
    indices_to_drop = []
    for run_key, grp in tqdm(df.groupby("_run", sort=False), desc="Checking .npy paths", unit="dir"):
        dir_path = base_path / run_key
        if not dir_path.is_dir():
            indices_to_drop.extend(grp.index.tolist())
            continue
        existing = {f.name for f in dir_path.iterdir() if f.suffix == ".npy"}
        for idx, row in grp.iterrows():
            if f"{row['tar_id']}.npy" not in existing:
                indices_to_drop.append(idx)
    df = df.drop(columns=["_run"]).drop(index=indices_to_drop).copy()
    n_after_npy = len(df)
    print(f"Dropped {n_after_frames - n_after_npy} rows (missing .npy)")
    print(f"Filtered total: {n_after_npy} rows (from {n_orig})")

    # Unique people after filtering
    persons = df["_person_id"].dropna().unique()
    persons = [p for p in persons if p]
    n_persons = len(persons)
    print(f"Unique people (run_path prefixes): {n_persons}")

    # 20% test persons, 80% train persons
    rng = random.Random(args.seed)
    persons_shuffled = persons.copy()
    rng.shuffle(persons_shuffled)
    n_test = max(1, int(n_persons * args.test_frac))
    test_persons = set(persons_shuffled[:n_test])
    train_persons = set(persons_shuffled[n_test:])

    df_test = df[df["_person_id"].isin(test_persons)].drop(columns=["_person_id"])
    df_train = df[df["_person_id"].isin(train_persons)].drop(columns=["_person_id"])

    print(f"Test: {len(test_persons)} people, {len(df_test)} rows")
    print(f"Train: {len(train_persons)} people, {len(df_train)} rows")

    # Output paths: same dir, use --output-stem for base name
    stem = args.output_stem
    out_dir = path.parent
    train_path = out_dir / f"{stem}_train.pkl"
    test_path = out_dir / f"{stem}_test.pkl"

    df_train.to_pickle(train_path)
    df_test.to_pickle(test_path)
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
