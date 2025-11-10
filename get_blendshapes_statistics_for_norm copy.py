from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from handle_split import get_side_from_db_parallel, transfer_attrs
from itertools import islice
from handle_split import load_dataframe, unique_runpaths
import random
import plotly.io as pio
import plotly.express as px
pio.renderers.default = 'browser'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

ALL_BLENDSHAPES_NAMES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 
                         'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                         'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 
                         'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
                         'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

def percentile_norm(data, percentile_low=2, percentile_high=98):
    low, high = np.percentile(data, [percentile_low, percentile_high])
    # data[data<low] = low
    # data[data>high] = high

    data = (data-low)/(high-low)
    return data, low, high

def hist_labels(matrix):
    values = matrix.flatten()
    import plotly.express as px
    # Create histogram
    fig = px.histogram(values, nbins=500, title="Histogram of normalized_labels")
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Count",
        bargap=0.1
    )
    pio.renderers.default = "browser"  # open in your default browser
    fig.show()
    
def get_lmk_statistics(lmks_lst):
    stats = {
        "position": {},
        "derivative": {}
    }

    # Get the number of channels from the first landmark data
    if len(lmks_lst) > 0:
        num_channels = lmks_lst[0].shape[1]
        channels = [f"ch_{i}" for i in range(num_channels)]
    else:
        channels = []

    # Initialize stat categories for each channel
    stat_types = ['min', 'max']#, '95%_min', '95%_max']
    for stat_type in stat_types:
        for kind in ['position', 'derivative']:
            if kind == 'position':
                stats[kind][stat_type] = {ch: [] for ch in channels}
            elif kind == 'derivative':
                stats[kind][stat_type] = {ch: [] for ch in channels}

    for lmks in lmks_lst:
        # Flatten positions - now working with (T, C) shape
        flat = lmks
        # percentiles = np.percentile(flat, [2.5, 97.5], axis=0)

        for i, ch in enumerate(channels):
            stats["position"]["min"][ch].append(np.min(flat[:, i]))
            stats["position"]["max"][ch].append(np.max(flat[:, i]))
            # stats["position"]["95%_min"][ch].append(percentiles[0, i])
            # stats["position"]["95%_max"][ch].append(percentiles[1, i])

        # Derivatives
        diff = np.diff(lmks, axis=0)
        # diff_percentiles = np.percentile(flat_diff, [2.5, 97.5], axis=0)

        for i, ch in enumerate(channels):
            stats["derivative"]["min"][ch].append(np.min(diff[:, i]))
            stats["derivative"]["max"][ch].append(np.max(diff[:, i]))


    # Convert lists to arrays for easier processing later
    for kind in stats:
        for stat_type in stats[kind]:
            for ch in channels:
                stats[kind][stat_type][ch] = np.array(stats[kind][stat_type][ch])

    return stats

def print_lmk_stats_summary(stats):
    print("🔍 Landmark Statistics Summary:\n")
    for kind in ['position', 'derivative']:
        print(f"=== {kind.upper()} ===")
        for stat_type in ['min', 'max', '95%_min', '95%_max']:
            print(f"\n-- {stat_type} --")
            # Get all channels for this kind and stat_type
            if kind in stats and stat_type in stats[kind]:
                channels = list(stats[kind][stat_type].keys())
                for ch in channels:
                    values = stats[kind][stat_type][ch]
                    print(f"  {ch}: mean = {values.mean(): .4f}, min = {values.min(): .4f}, max = {values.max(): .4f}")
        print("\n" + "-" * 40 + "\n")

def get_channel_bounds_vector_from_data(lmks_lst):
    """
    Create a vector where each channel has its (0.05% min, 95% max) values tuple
    directly from the raw data.

    Args:
        lmks_lst: List of numpy arrays with shape (T, C)

    Returns:
        List of tuples: [(ch_0_min, ch_0_max), (ch_1_min, ch_1_max), ...]
    """
    if len(lmks_lst) == 0:
        return []

    # Stack all data to get overall statistics
    all_data = np.vstack(lmks_lst)  # Shape: (total_T, C)

    # Calculate percentiles for each channel
    percentiles = np.percentile(all_data, [0.05, 95], axis=0)

    # Create bounds vector
    bounds_vector = []
    for i in range(all_data.shape[1]):
        min_val = percentiles[0, i]  # 0.05% percentile
        max_val = percentiles[1, i]  # 95% percentile
        bounds_vector.append((min_val, max_val))

    return bounds_vector

def print_channel_bounds_from_data(lmks_lst):
    """Print the channel bounds (0.05% min, 95% max) directly from raw data."""
    bounds = get_channel_bounds_vector_from_data(lmks_lst)

    print("📊 Channel Bounds (0.05% min, 95% max) from Raw Data:\n")
    for i, (min_val, max_val) in enumerate(bounds):
        print(f"  ch_{i}: ({min_val:.4f}, {max_val:.4f})")
    print()

    return bounds

def find_uniform_transformation(data, method='histogram_equalization', n_bins=100, return_inverse=False):

    import scipy.stats as stats
    from scipy.interpolate import interp1d
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    transformed_data = np.zeros_like(data)
    transform_functions = []
    inverse_functions = []
    

    if method == 'histogram_equalization':
        # Histogram equalization using CDF
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        cdf = np.cumsum(hist) * (bin_edges[1] - bin_edges[0])
        cdf = np.insert(cdf, 0, 0)  # Add 0 at the beginning
        
        # Create interpolation function for the transformation
        transform_func = interp1d(bin_edges, cdf, bounds_error=False, fill_value=(0, 1))
        
        # Apply transformation
        transformed_data = transform_func(data)
        
        # Create inverse transformation
        if return_inverse:
            # Use quantile-based inverse
            sorted_indices = np.argsort(data)
            inverse_func = interp1d(
                np.linspace(0, 1, len(data)),
                data[sorted_indices],
                bounds_error=False,
                fill_value=(data.min(), data.max())
            )
            inverse_functions.append(inverse_func)
        
    elif method == 'quantile':
        # Quantile transformation using empirical CDF
        sorted_data = np.sort(data)
        ranks = np.searchsorted(sorted_data, data, side='left')
        transformed_data = ranks / len(data)
        
        # Create transformation function
        def transform_func(x):
            ranks = np.searchsorted(sorted_data, x, side='left')
            return np.clip(ranks / len(sorted_data), 0, 1)
        
        transform_functions.append(transform_func)
        
        # Create inverse transformation
        if return_inverse:
            def inverse_func(x):
                indices = np.clip(x * len(sorted_data), 0, len(sorted_data) - 1).astype(int)
                return sorted_data[indices]
            inverse_functions.append(inverse_func)
            
    elif method == 'power':
        # Find optimal power transformation using Box-Cox or Yeo-Johnson
        try:
            # Try Box-Cox transformation (requires positive data)
            if np.all(data > 0):
                transformed_data, lambda_param = stats.boxcox(data)
                # Normalize to [0, 1]
                transformed_data = (transformed_data - transformed_data.min()) / (transformed_data.max() - transformed_data.min())
                
                def transform_func(x):
                    if np.all(x > 0):
                        transformed = stats.boxcox(x, lambda_param)
                        return (transformed - transformed_data.min()) / (transformed_data.max() - transformed_data.min())
                    else:
                        return x  # Return original if transformation not possible
                
                transform_functions.append(transform_func)
                
                if return_inverse:
                    def inverse_func(x):
                        # Denormalize and apply inverse Box-Cox
                        denorm = x * (transformed_data.max() - transformed_data.min()) + transformed_data.min()
                        return stats.invboxcox(denorm, lambda_param)
                    inverse_functions.append(inverse_func)
            else:
                # Use Yeo-Johnson for data that can be negative
                transformed_data, lambda_param = stats.yeojohnson(data)
                transformed_data = (transformed_data - transformed_data.min()) / (transformed_data.max() - transformed_data.min())
                
                def transform_func(x):
                    transformed = stats.yeojohnson(x, lambda_param)
                    return (transformed - transformed_data.min()) / (transformed_data.max() - transformed_data.min())
                
                transform_functions.append(transform_func)
                
                if return_inverse:
                    def inverse_func(x):
                        denorm = x * (transformed_data.max() - transformed_data.min()) + transformed_data.min()
                        return stats.invyeojohnson(denorm, lambda_param)
                    inverse_functions.append(inverse_func)
                    
        except:
            # Fallback to simple power transformation
            power = 0.5  # Square root
            transformed_data = np.power(np.abs(data), power)
            transformed_data = (transformed_data - transformed_data.min()) / (transformed_data.max() - transformed_data.min())
            
            def transform_func(x):
                transformed = np.power(np.abs(x), power)
                return (transformed - transformed_data.min()) / (transformed_data.max() - transformed_data.min())
            
            transform_functions.append(transform_func)
            
            if return_inverse:
                def inverse_func(x):
                    denorm = x * (transformed_data.max() - transformed_data.min()) + transformed_data.min()
                    return np.power(denorm, 1/power)
                inverse_functions.append(inverse_func)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'histogram_equalization', 'quantile', or 'power'")
        
    
    return transformed_data


data_path = Path("/mnt/ML/Development/shaked.dovrat/blendshapes_fairseq")
split_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
df = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes_cleaned2_with_side.pkl" #"4M_20250220_loud_valid_lmks.pkl"
df_path = split_path / df
df = load_dataframe(df_path)
run_paths = unique_runpaths(df)
sampled_run_paths = random.sample(run_paths, k=10000)

lmks_lst = []
all_diffs = []
for run_path in tqdm(sampled_run_paths):
    lmks = np.load(data_path / run_path / "landmarks_and_blendshapes.npy")
    diff = np.diff(lmks, axis=0)
    lmks_lst.append(lmks)
    all_diffs.append(diff)
    
lmks_stacked = np.concatenate(lmks_lst, axis=0) 
lmks_diff_stacked = np.concatenate(all_diffs, axis=0) 

boundaries_lmks = []
boundaries_diffs = []
for i in range(52):
    _, low, high = percentile_norm(lmks_stacked[:, i], percentile_low=0, percentile_high=95)
    boundaries_lmks.append((low, high))
    _, low, high = percentile_norm(lmks_diff_stacked[:, i], percentile_low=2, percentile_high=98)
    boundaries_diffs.append((low, high))

print("="*100)
print("boundaries_lmks")
print(boundaries_lmks)
print("="*100)
print("boundaries_diffs")
print(boundaries_diffs)

# rows, cols = 13, 4  # 13*4 = 52 subplots

# fig = make_subplots(rows=rows, cols=cols,subplot_titles=[f"{ALL_BLENDSHAPES_NAMES[i]}" for i in range(52)])

# for i in range(52):
#     r, c = divmod(i, cols)
#     fig.add_trace(
#         go.Histogram(x=percentile_norm(lmks_stacked[:, i],percentile_low=0, percentile_high=95), nbinsx=500, name=f"{ALL_BLENDSHAPES_NAMES[i]}", showlegend=False),
#         # go.Histogram(x=np.log(1+100*lmks_stacked[:, i]), nbinsx=50, name=f"{ALL_BLENDSHAPES_NAMES[i]}", showlegend=False),
#         row=r+1, col=c+1)

# fig.update_layout(
#     height=3000, width=1200,
#     title_text="Histograms of percentile normalized blendshapes no cliping, percentile_low=0, percentile_high=95")
# fig.show()

                               
# rows, cols = 13, 4  # 13*4 = 52 subplots

# fig = make_subplots(rows=rows, cols=cols,subplot_titles=[f"{ALL_BLENDSHAPES_NAMES[i]}" for i in range(52)])

# for i in range(52):
#     r, c = divmod(i, cols)
#     fig.add_trace(
#         go.Histogram(x=percentile_norm(lmks_diff_stacked[:, i]), nbinsx=500, name=f"{ALL_BLENDSHAPES_NAMES[i]}", showlegend=False),
#         row=r+1, col=c+1)

# fig.update_layout(
#     height=3000, width=1200,
#     title_text="Histograms of percentile normalized diffs blendshapes, 2, 98")

# fig.show()


# # rows, cols = 13, 4  # 13*4 = 52 subplots

# # fig = make_subplots(rows=rows, cols=cols,subplot_titles=[f"{ALL_BLENDSHAPES_NAMES[i]}" for i in range(52)])

# # for i in range(52):
# #     r, c = divmod(i, cols)
# #     fig.add_trace(
# #         go.Histogram(x=find_uniform_transformation(lmks_stacked[:, i]), nbinsx=50, name=f"{ALL_BLENDSHAPES_NAMES[i]}", showlegend=False),
# #         # go.Histogram(x=np.log(1+100*lmks_stacked[:, i]), nbinsx=50, name=f"{ALL_BLENDSHAPES_NAMES[i]}", showlegend=False),
# #         row=r+1, col=c+1)

# # fig.update_layout(
# #     height=3000, width=1200,
# #     title_text="Histograms of histogram normalized blendshapes")
# # fig.show()






# # rows, cols = 13, 4  # 13*4 = 52 subplots

# # fig = make_subplots(rows=rows, cols=cols,subplot_titles=[f"{ALL_BLENDSHAPES_NAMES[i]}" for i in range(52)])

# # for i in range(52):
# #     r, c = divmod(i, cols)
# #     fig.add_trace(
# #         go.Histogram(x=np.power(lmks_stacked[:, i], 0.5), nbinsx=50, name=f"{ALL_BLENDSHAPES_NAMES[i]}", showlegend=False),
# #         # go.Histogram(x=np.log(1+100*lmks_stacked[:, i]), nbinsx=50, name=f"{ALL_BLENDSHAPES_NAMES[i]}", showlegend=False),
# #         row=r+1, col=c+1)

# # fig.update_layout(
# #     height=3000, width=1200,
# #     title_text="Histograms of all 52 columns")
# # import plotly.io as pio
# # fig.show()


# diff_stacked = np.concatenate(all_diffs, axis=0)

# min_lmks =  np.min(lmks_stacked, axis=0)
# max_lmks =  np.max(lmks_stacked, axis=0)
# min_diffs =  np.min(diff_stacked, axis=0)
# max_diffs =  np.max(diff_stacked, axis=0)

# i = 50
# norm_lmks = (lmks_lst[i] - min_lmks) / (max_lmks - min_lmks)
# hist_labels(lmks_lst[i])
# hist_labels(norm_lmks)





# for lmks in tqdm(lmks_lst):
#     diff = np.diff(lmks, axis=0)  # shape: (T-1, L, 3)
    

# # Stack all differences across all videos
# all_diffs_stacked = np.vstack(all_diffs)  # shape: (total_points, 3)



# # Alternative: use the print function

# # Get channel bounds vector directly from raw data
# print("=" * 60)
# channel_bounds = get_channel_bounds_vector_from_data(lmks_lst)
# print(f"Channel bounds vector length: {len(channel_bounds)}")

# print_channel_bounds_from_data(lmks_lst)

# ##########################################################
# # Get channel bounds vector directly from raw data
# print("=" * 60)
# print("Channel diffs bounds")
# channel_diffs_bounds = get_channel_bounds_vector_from_data(all_diffs)
# print(f"Channel diffs bounds vector length: {len(channel_diffs_bounds)}")
# print_channel_bounds_from_data(all_diffs)
# a=1