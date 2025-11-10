from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from handle_split import get_side_from_db_parallel, transfer_attrs
from itertools import islice
from handle_split import load_dataframe, unique_runpaths
import random

idx = [37, 39, 40, 42, 61, 78, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 183, 185, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246, 162, 67, 132, 103, 136, 234, 172, 109, 176, 148, 149, 150, 21, 54, 58, 93, 127, 468, 469, 470, 471, 472]
def get_lmk_statistics(lmks_lst):
    axes = ['x', 'y', 'z']
    stats = {
        "position": {},
        "derivative": {}
    }

    # Initialize stat categories for each axis
    stat_types = ['min', 'max']#, '95%_min', '95%_max']
    for stat_type in stat_types:
        for kind in ['position', 'derivative']:
            if kind == 'position':
                stats[kind][stat_type] = {axis: [] for axis in axes}
            elif kind == 'derivative':
                stats[kind][stat_type] = {axis: [] for axis in axes + ['v']}
            # stats[kind][stat_type] = {axis: [] for axis in axes}

    for lmks in lmks_lst:
        # Flatten positions
        flat = lmks.reshape(-1, 3)
        # percentiles = np.percentile(flat, [2.5, 97.5], axis=0)

        for i, axis in enumerate(axes):
            stats["position"]["min"][axis].append(np.min(flat[:, i]))
            stats["position"]["max"][axis].append(np.max(flat[:, i]))
            # stats["position"]["95%_min"][axis].append(percentiles[0, i])
            # stats["position"]["95%_max"][axis].append(percentiles[1, i])

        # Derivatives
        diff = np.diff(lmks, axis=0)
        flat_diff = diff.reshape(-1, 3)
        abs_v = np.linalg.norm(diff, axis=2, keepdims=True)
        # diff_percentiles = np.percentile(flat_diff, [2.5, 97.5], axis=0)

        for i, axis in enumerate(axes):
            stats["derivative"]["min"][axis].append(np.min(flat_diff[:, i]))
            stats["derivative"]["max"][axis].append(np.max(flat_diff[:, i]))
            # stats["derivative"]["95%_min"][axis].append(diff_percentiles[0, i])
            # stats["derivative"]["95%_max"][axis].append(diff_percentiles[1, i])
        stats["derivative"]["min"]['v'].append(np.min(abs_v))
        stats["derivative"]["max"]['v'].append(np.max(abs_v))

    # Convert lists to arrays for easier processing later
    for kind in stats:
        for stat_type in stats[kind]:
            for axis in axes:
                stats[kind][stat_type][axis] = np.array(stats[kind][stat_type][axis])
    
    for stat_type in stats[kind]:
        stats["derivative"][stat_type]['v'] = np.array(stats["derivative"][stat_type]['v'])

    return stats

def print_lmk_stats_summary(stats):
    print("🔍 Landmark Statistics Summary:\n")
    for kind in ['position', 'derivative']:
        print(f"=== {kind.upper()} ===")
        for stat_type in ['min', 'max', '95%_min', '95%_max']:
            print(f"\n-- {stat_type} --")
            for axis in ['x', 'y', 'z']:
                values = stats[kind][stat_type][axis]
                print(f"  {axis.upper()}: mean = {values.mean(): .4f}, min = {values.min(): .4f}, max = {values.max(): .4f}")
        print("\n" + "-" * 40 + "\n")
        
data_path = Path("/mnt/ML/Development/katya.ivantsiv/landmarks")
split_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
df = "loud_and_whisper_and_lip_20250713_064722.pkl" #"4M_20250220_loud_valid_lmks.pkl"
df_path = split_path / df
df = load_dataframe(df_path)
run_paths = unique_runpaths(df)
sampled_run_paths = random.sample(run_paths, k=1000)

lmks_lst = []
for run_path in tqdm(sampled_run_paths):
    lmks = np.load(data_path / run_path / "landmarks.npy")
    lmks_lst.append(lmks[:,idx,:])

all_diffs = []

for lmks in tqdm(lmks_lst):
    diff = np.diff(lmks, axis=0)  # shape: (T-1, L, 3)
    flat_diff = diff.reshape(-1, 3)
    abs_v = np.linalg.norm(flat_diff, axis=1, keepdims=True)# shape: ((T-1)*L, 3)
    all_diffs.append(np.concatenate((flat_diff, abs_v),axis=1))

# Stack all differences across all videos
all_diffs_stacked = np.vstack(all_diffs)  # shape: (total_points, 3)

# Compute mean and std for x, y, z
mean_xyz = np.mean(all_diffs_stacked, axis=0)
std_xyz = np.std(all_diffs_stacked, axis=0)

# Print results
for axis, mean, std in zip(['x', 'y', 'z','v'], mean_xyz, std_xyz):
    print(f"{axis}: mean = {mean:.6f}, std = {std:.6f}")



##########################################################
# lmks_mt = np.stack(lmks_lst)
stats = get_lmk_statistics(lmks_lst)
for kind in ['position', 'derivative']:
    print(f"\n--- {kind.upper()} ---")
    for stat_type in ['min', 'max']:
        print(f"\n{stat_type.upper()}:")
        for axis, values in stats[kind][stat_type].items():
            values = np.array(values)
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            print(f"  {axis}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
            
            
##########################################################
for kind in ['position', 'derivative']:
    print(f"\n--- {kind.upper()} ---")
    for stat_type in ['min', 'max']:
        print(f"\n{stat_type.upper()}:")
        for axis, values in stats[kind][stat_type].items():
            values = np.array(values)
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            std_val = np.std(values)

            # 95% confidence interval (assuming normal dist)
            ci_low = mean_val - 1.96 * std_val
            ci_high = mean_val + 1.96 * std_val

            # 2.5 and 97.5 percentiles (robust 95% range)
            p2_5 = np.percentile(values, 2.5)
            p97_5 = np.percentile(values, 97.5)

            print(f"  {axis}:")
            print(f"    min     = {min_val:.4f}")
            print(f"    max     = {max_val:.4f}")
            print(f"    mean    = {mean_val:.4f}")
            print(f"    std     = {std_val:.4f}")
            print(f"    mean ±1.96×std (95% CI) = [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"    95% range (percentiles) = [{p2_5:.4f}, {p97_5:.4f}]")
a = 1