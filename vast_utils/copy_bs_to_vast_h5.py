from pathlib import Path
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from files_utils import safe_save

user_name = "katya.ivantsiv"

# Global variables for worker processes (initialized via initializer)
_worker_path_to_tar_id = None
_worker_origin_paths = None
_worker_destination_folder = None
_worker_q_features_paths = None


def _init_worker(path_to_tar_id, origin_paths, destination_folder, q_features_paths):
    """Initialize worker process with shared data."""
    global _worker_path_to_tar_id, _worker_origin_paths, _worker_destination_folder, _worker_q_features_paths
    _worker_path_to_tar_id = path_to_tar_id
    _worker_origin_paths = origin_paths
    _worker_destination_folder = destination_folder
    _worker_q_features_paths = q_features_paths


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


origin_paths_to_copy_from = [Path("/mnt/A3000/Recordings/v2_data")]
destination_folder = Path("/mnt/ML/Development/katya.ivantsiv/blendshapes")
# splits_base_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
# split_filename = "loud_and_whisper_and_lip_20250713_064722.pkl" 
# splits_base_path = Path("/mnt/ML/Development/ML_Data_DB/v2/splits/full/20251106_split_1")
# split_filename = "kfold_train_251106_no_bookkeeping_r8_mid_th_sil_0_5_lip_0_4_whi_0_34_loud_0_167.pkl"
# splits_base_path = Path("/mnt/ML/Development/ML_Data_DB/v2/splits/full/20251106_split_1")
# split_filename = "kfold_train_251106_no_bookkeeping_r8_mid_th_sil_0_5_lip_0_4_whi_0_34_loud_0_167.pkl"
splits_base_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
split_filename = "expressive_blendshapes_no_glasses_110326_train.pkl"

# split_filename = "train.pkl"

q_features_paths = [
    Path(os.environ.get("Q_FEATURES_PATH_1", "/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")),
    Path(os.environ.get("Q_FEATURES_PATH_2", "/mnt/A3000/Recordings/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")),
]
# q_features_path = Path("/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")

# HDF5 split file
# split_full_path = Path("/mnt/ML/ModelsTrainResults/angelina.heyler/multimodal_nemo/splits/clearvoice_brainvoice_intersection_split_with_side_and_sizes/kfold_train_251106_no_bookkeeping_r9_mid_th_sil_0_5_lip_0_4_whi_0_34_loud_0_167_with_side.h5")
split_full_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/blendshapes_KatyaIvantsiv_2026_01_11.h5")
print(f"Loading split file: {split_full_path}")
df = pd.read_hdf(split_full_path)

# Create a lightweight lookup dict instead of passing entire DataFrame to workers
# This is MUCH more memory efficient: dict is copied once per worker, not the whole DataFrame
print("Creating path -> tar_id lookup...")
path_to_tar_id = df.groupby("run_path")["tar_id"].first().to_dict()

unique_paths = list(path_to_tar_id.keys())
unique_paths = sorted(unique_paths)
print(f"processing {len(unique_paths)} paths in split {split_filename}")

# Free the DataFrame memory - we only need the lookup dict now
del df

def process_single_path(path):
    """Process a single path and return validation results.
    
    Uses global worker variables initialized by _init_worker to avoid
    copying large data structures to each worker.
    """
    result = {
        'success': 0,
        'already_exists': 0,
        'path_not_exist': 0,
        'not_loadable': 0,
        'features_path_not_exist': 0,
        'has_nans': 0,
        'lengths_mismatch': 0,
        'values_out_of_range': 0
    }
    
    # Use worker-local globals (set via initializer)
    tar_id = _worker_path_to_tar_id.get(path)
    if tar_id is None:
        result['path_not_exist'] = 1
        return result
    
    destination_path = _worker_destination_folder / path / "blendshapes.npy"
    # Resolve features path: try each candidate base until one exists
    features_path = None
    for base in _worker_q_features_paths:
        candidate = base / path / f"{tar_id}.npy"
        if candidate.exists():
            features_path = candidate
            break
    if features_path is None:
        features_path = _worker_q_features_paths[0] / path / f"{tar_id}.npy"
    
    # If file is already in the destination and valid - skip
    # is_valid, error, bs = check_bs_validity(destination_path, features_path)
    is_valid = destination_path.exists()
    if is_valid:
        result['success'] = 1
        result['already_exists'] = 1
        return result
    # elif error != "path_not_exist":
    #     destination_path.unlink()

    # # If file is in another folder in the vast, and valid - copy from there
    # bs_vast_path = _worker_origin_paths[0] / path / "landmarks_and_blendshapes.npy"
    # is_valid, error, bs = check_bs_validity(bs_vast_path, features_path)
    # if is_valid:
    #     safe_save(bs, destination_path, "float16")
    #     result['success'] = 1
    #     return result
    # elif error != "path_not_exist":
    #     result[error] = 1

    # check validity and copy from the a3000
    bs_a3000_path = _worker_origin_paths[0] / path / "landmarks_and_blendshapes.npz"
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

# Use multiprocessing Pool with initializer to share data efficiently
# The initializer runs once per worker, avoiding copying data with each task
with mp.Pool(
    processes=n_jobs,
    initializer=_init_worker,
    initargs=(path_to_tar_id, origin_paths_to_copy_from, destination_folder, q_features_paths)
) as pool:
    with tqdm(total=len(unique_paths), desc="Processing files", unit="file", mininterval=0.5) as pbar:
        # Use imap_unordered for slightly better performance (no need to maintain order)
        # Optimal chunksize for 19M files: balance between overhead and memory
        optimal_chunksize = 5000
        print(f"Using chunksize: {optimal_chunksize}")
        
        for res in pool.imap_unordered(process_single_path, unique_paths, chunksize=optimal_chunksize):
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
    percentage = (value / total_files) * 100
    if key == 'success':
        print(f"   ✅ {key.capitalize()}: {value}, {percentage:.2f}%")
    elif key == 'already_exists':
        print(f"   📦 {key.replace('_', ' ').title()}: {value}, {percentage:.2f}%")
    else:
        print(f"   ❌ {key.replace('_', ' ').title()}: {value}, {percentage:.2f}%")
print("=" * 80)

# Save summary to file
summaries_folder = Path("/home/katya.ivantsiv/utils_scripts/vast_utils/copy_summaries")
summaries_folder.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = summaries_folder / f"copy_summary_{timestamp}.json"

summary_data = {
    "timestamp": timestamp,
    "split_file": str(split_full_path),
    "destination_folder": str(destination_folder),
    "unique_paths_length": len(unique_paths),
    "total_files": total_files,
    "results": {key: f"{value}, {(value / total_files) * 100:.2f}%" for key, value in counter.items()}
}

with open(summary_file, "w") as f:
    json.dump(summary_data, f, indent=2)

print(f"\n📁 Summary saved to: {summary_file}")
            
