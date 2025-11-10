from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths
vast_path = Path('/mnt/ML/Development/katya.ivantsiv/landmarks')
df_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
filename = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes.pkl"
full_path = df_path / filename

# Load original DataFrame
print("Loading DataFrame from pickle...")
df = pd.read_pickle(full_path)
print(f"Loaded DataFrame with {len(df)} rows.")

# Get unique run_paths
unique_run_paths = df["run_path"].unique()

# Keep track of valid and invalid run_paths
valid_run_paths = set()
deleted_files_count = 0

# Check each run_path only once
for run_path in tqdm(unique_run_paths):
    npy_path = vast_path / run_path / "landmarks.npy"
    if npy_path.exists():
        try:
            _ = np.load(npy_path)
            valid_run_paths.add(run_path)
        except Exception as e:
            print(f"Failed to load {npy_path}: {e}")
            try:
                npy_path.unlink()
                print(f"Deleted {npy_path}, so far deleted {deleted_files_count} files")
                deleted_files_count += 1
            except Exception as delete_err:
                print(f"Failed to delete {npy_path}: {delete_err}")

# Filter and retain full rows from original df
new_df = df[df["run_path"].isin(valid_run_paths)].reset_index(drop=True)

new_df.attrs = df.attrs

# Save new DataFrame
new_path = df_path / '4M_20250220_loud_valid_lmks.pkl'
new_df.to_pickle(new_path)

print(f"Saved new DataFrame with {len(new_df)} rows to {new_path}")
print(f"Deleted {deleted_files_count} invalid .npy files.")
