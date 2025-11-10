from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

from files_utils import safe_save

user_name = "katya.ivantsiv"


# we want to train on the landmarks split
# check if path exists in the destination.
# if it exists there check if its valid
# for each path - if it exists in the vast copy it from the vast
# if not - copy from the a3000
# before copying checking file validity:
# run the following tests:
# 0. the file is being able to be loaded
# 1. the file has no nans
# 2. the length of the file test (as the q featurs length)
# 3. the values of the numbers are between 0 and 1
# if all test are successful - save the file as float 16
# if not, save into a log file what test failed

def get_npy_shape(filename):
    with open(filename, "rb") as f:
        # Read the magic string and version
        magic = f.read(6)
        if magic[:6] != b"\x93NUMPY":
            raise ValueError("Not a valid .npy file")

        # Read version
        version = f.read(2)
        major, minor = version

        # Read the header length
        if major == 1:
            header_len = np.frombuffer(f.read(2), dtype=np.uint16)[0]
        elif major == 2:
            header_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        else:
            raise ValueError("Unsupported .npy file version")

        # Read the header
        header = f.read(header_len).decode("latin1")

        # Extract the shape from the header
        header_dict = eval(header)
        return header_dict["shape"]

def process_run_path(run_path, blendshapes_dirs, out_dir):
    run_path = Path(run_path)
    blendshapes_rel_path = run_path / "landmarks_and_blendshapes.npz"

    for blendshapes_dir in blendshapes_dirs:
        blendshapes_full_path = blendshapes_dir / blendshapes_rel_path
        if blendshapes_full_path.exists():
            blendshapes_out_rel_path = blendshapes_rel_path.with_suffix(".npy")
            blendshapes_out_path = out_dir / blendshapes_out_rel_path
            try:
                if blendshapes_out_path.exists():
                    try:
                        blendshapes_length = get_npy_shape(blendshapes_out_path)[0]
                    except Exception:
                        blendshapes = np.load(blendshapes_full_path)["blendshapes"]
                        blendshapes_out_path.parent.mkdir(parents=True, exist_ok=True)
                        safe_save(blendshapes, blendshapes_out_path)
                        blendshapes_length = blendshapes.shape[0]
                else:
                    blendshapes = np.load(blendshapes_full_path)["blendshapes"]
                    blendshapes_out_path.parent.mkdir(parents=True, exist_ok=True)
                    safe_save(blendshapes, blendshapes_out_path, "float16")
                    blendshapes_length = blendshapes.shape[0]
            except Exception as e:
                print(f"Error processing {blendshapes_full_path}: {e}")
                return run_path, None, np.nan

            return run_path, str(blendshapes_out_rel_path), blendshapes_length

    return str(run_path), None, np.nan


def check_bs_loadable(bs_path):
    """Check if the blendshape file can be loaded."""
    try:
        bs_lmk = np.load(bs_path)
        if isinstance(bs_lmk, np.lib.npyio.NpzFile):
            bs = bs_lmk['blendshapes']
        else:
            bs = bs_lmk
        return bs, True
    except Exception:
        return None, False

def check_bs_no_nans(bs):
    """Check if the blendshape array has no NaNs."""
    if np.isnan(bs).any():
        return False
    return True

def check_bs_vs_featurs_length(bs, features_path):
    """Check if the blendshape array has the same length as the features array."""
    features = np.load(features_path)
    if not features_path.exists():
        return False
    features_len = features.shape[0]
    bs_len = bs.shape[0]
    if np.abs(features_len * 30 / 200 - bs_len) > 15:
        return False
    return True

def check_bs_values_in_range(bs):
    """Check if all blendshape values are between 0 and 1."""
    if np.any((bs < 0) | (bs > 1)):
        return False
    return True

def check_bs_validity(bs_path, features_path):
    """Run all blendshape checks and return a dict of results."""
    if not bs_path.exists():
        return False, 'path_not_exist', None
    bs, is_valid = check_bs_loadable(bs_path)
    if not is_valid:
        return False, 'not_loadable', None
    if not check_bs_no_nans(bs):
        return False, 'has_nans', None
    if not features_path.exists():
        return False, 'features_path_not_exist', None
    
    if not check_bs_vs_featurs_length(bs, features_path):
        return False, 'lengths_mismatch', None
    if not check_bs_values_in_range(bs):
        return False, 'values_out_of_range', None
    return True, None, bs

def get_tar_id_from_path(path, df):
    tar_id = df[df["run_path"] == path]["tar_id"].iloc[0]
    return tar_id


origin_paths_to_copy_from = [Path("/mnt/ML/Development/shaked.dovrat/blendshapes_fairseq"),
                             Path("/mnt/A3000/Recordings/v2_data")]
destination_folder = Path("/mnt/ML/Development/katya.ivantsiv/blendshapes")
splits_base_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
# split_filename = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes_cleaned2_with_side_first_100_rows.pkl"
split_filename = "loud_and_whisper_and_lip_20250713_064722.pkl"

# split_filename = "WHISPER_GIP_general_clean_250415_v2_with_side_attrs_valid_20250709_190842.pkl"
# split_filename = "LIP_GIP_general_clean_250415_v2_with_side_attrs.pkl"

# split_filename = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes.pkl"
q_features_path = Path("/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")


split_full_path = splits_base_path / split_filename
df = pd.read_pickle(split_full_path)

unique_paths = df["run_path"].unique()
unique_paths = sorted(unique_paths)
print(f"processing {len(unique_paths)} paths in split {split_filename}")

def process_single_path(path):
    """Process a single path and return validation results."""
    result = {
        'success': 0,
        'path_not_exist': 0,
        'not_loadable': 0,
        'features_path_not_exist': 0,
        'has_nans': 0,
        'lengths_mismatch': 0,
        'values_out_of_range': 0
    }
    
    destination_path = destination_folder / path / "blendshapes.npy"
    features_path = q_features_path / path / f"{get_tar_id_from_path(path, df)}.npy"
    
    # If file is already in the destination and valid - skip
    # is_valid, error, bs = check_bs_validity(destination_path, features_path)
    is_valid = destination_path.exists()
    if is_valid:
        result['success'] = 1
        return result
    # elif error != "path_not_exist":
    #     destination_path.unlink()

    # If file is in another folder in the vast, and valid - copy from there
    bs_vast_path = origin_paths_to_copy_from[0] / path / "landmarks_and_blendshapes.npy"
    is_valid, error, bs = check_bs_validity(bs_vast_path, features_path)
    if is_valid:
        safe_save(bs, destination_path, "float16")
        result['success'] = 1
        return result
    elif error != "path_not_exist":
        result[error] = 1

    # check validity and copy from the a3000
    bs_a3000_path = origin_paths_to_copy_from[1] / path / "landmarks_and_blendshapes.npz"
    is_valid, error, bs = check_bs_validity(bs_a3000_path, features_path)
    if is_valid:
        safe_save(bs, destination_path, "float16")
        result['success'] = 1
        return result
    elif error:
        result[error] = 1
    
    return result

# Get number of available CPU cores
n_jobs = 96 #mp.cpu_count()  # Use all available cores

print(f"Starting parallel processing with {n_jobs} cores...")
unique_paths = list(reversed(unique_paths))
total_files = len(unique_paths)
print(f"total files: {total_files}")

counter = Counter()

# Use multiprocessing Pool with imap for better progress tracking
with mp.Pool(processes=n_jobs) as pool:
    with tqdm(total=len(unique_paths), desc="Processing files", unit="file", mininterval=0.5) as pbar:
        # Use imap to get results as they complete, maintaining order
        # Optimal chunksize for 7.5M files: total_files / (num_workers * 4)
        optimal_chunksize = 2000
        print(f"Using chunksize: {optimal_chunksize}")
        
        for res in pool.imap(process_single_path, unique_paths, chunksize=optimal_chunksize):
            # Update cumulative counters
            counter.update(res)

            # Update tqdm display
            pbar.update(1)
            pbar.set_postfix(dict(counter))
            


# Print final summary
print("\n" + "=" * 80)
print("🎉 PROCESSING COMPLETE!")
print("=" * 80)
print("📈 Final Results:")
for key, value in counter.items():
    if key == 'success':
        print(f"   ✅ {key.capitalize()}: {value}")
    else:
        print(f"   ❌ {key.replace('_', ' ').title()}: {value}")
print("=" * 80)
            
