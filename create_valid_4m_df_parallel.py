from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Paths
vast_path = Path('/mnt/ML/Development/katya.ivantsiv/landmarks')
df_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
filename = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes_cleaned2_with_side.pkl"
full_path = df_path / filename

# Load DataFrame
print("Loading DataFrame from pickle...")
df = pd.read_pickle(full_path)
# df = df.head(290920)
print(f"Loaded DataFrame with {len(df)} rows.")


# Unique run paths
unique_run_paths = df["run_path"].unique()

# Function to check a single run_path
def check_run_path(run_path):
    npy_path = vast_path / run_path / "landmarks.npy"
    if npy_path.exists():
        try:
            array = np.load(npy_path)
            has_nan = np.isnan(array).any()
            return run_path, has_nan #True
        except Exception:
            try:
                npy_path.unlink()
                return run_path, False
            except Exception:
                return run_path, False
    return run_path, False

# Parallel processing
print("Checking landmarks.npy in parallel...")
# results = Parallel(n_jobs=1)(delayed(check_run_path)(rp) for rp in unique_run_paths)
results = Parallel(n_jobs=50)(delayed(check_run_path)(rp) for rp in tqdm(unique_run_paths, desc="Checking landmarks"))

valid_run_paths = {rp for rp, is_valid in results if is_valid}
# deleted_files_count = sum(1 for _, is_valid in results if not is_valid)

# # Filter DataFrame
# new_df = df[df["run_path"].isin(valid_run_paths)].reset_index(drop=True)

# # Save filtered DataFrame
# new_path = df_path / '4M_20250220_loud_valid_lmks.pkl'
# new_df.to_pickle(new_path)

# Summary
# print(f"✅ Saved new DataFrame with {len(new_df)} rows to {new_path}")
# print(f"🗑️  Deleted {deleted_files_count} invalid or broken landmarks.npy files.")
print(f"  Nan paths {len(valid_run_paths)} invalid or broken landmarks.npy files.")
