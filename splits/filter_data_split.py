
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Load the original pickle
print("Loading DataFrame from pickle...")
path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
filename = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes.pkl"
full_path = path / filename
df = pd.read_pickle(full_path)
print(f"Loaded DataFrame with {len(df)} rows.")

# Set data root path
blendshapes_path = Path("/mnt/ML/Development/katya.ivantsiv/blendshapes")
q_features_path = Path("/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")

# Function to validate a single row
def validate_row(args):
    index, row = args
    try:
        tar_id = row['tar_id']
        run_path = row['run_path']
        features_path = q_features_path / run_path / f'{tar_id}.npy'
        landmarks_and_blendshapes_path = blendshapes_path / run_path / 'landmarks_and_blendshapes.npz'

        if not features_path.exists() or not landmarks_and_blendshapes_path.exists():
            return index  # drop if any file is missing

        landmarks_and_blendshapes = np.load(landmarks_and_blendshapes_path)
        blendshape_label = landmarks_and_blendshapes['blendshapes']
        bs_len = blendshape_label.shape[0]

        features = np.load(features_path, mmap_mode='r')
        features_len = features.shape[0]

        if np.abs(features_len * 30 / 200 - bs_len) > 15:
            return index

        if np.any((blendshape_label < 0) | (blendshape_label > 1) | np.isnan(blendshape_label)):
            return index

        return None  # valid row
    except Exception as e:
        # If any error occurs, mark for dropping
        return index

# Use multiprocessing for speed
print("Validating rows in parallel...")
with Pool(cpu_count()) as pool:
    # results = pool.map(validate_row, df.iterrows())
    results = list(tqdm(pool.imap(validate_row, df.iterrows()), total=len(df), desc="Validating"))


# Collect indices to drop
indices_to_drop = [index for index in results if index is not None]
print(f"Dropping {len(indices_to_drop)} invalid rows...")

# Drop and reset
df_cleaned = df.drop(index=indices_to_drop).reset_index(drop=True)

# Save the cleaned DataFrame
cleaned_filename = filename.replace(".pkl", "_cleaned2.pkl")
df_cleaned.to_pickle(path / cleaned_filename)
print(f"Saved cleaned DataFrame with {len(df_cleaned)} rows to {cleaned_filename}")



# from pathlib import Path
# import pandas as pd
# import numpy as np

# # Load the original pickle
# print("Loading DataFrame from pickle...")
# path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
# filename = "LOUD_GIP_general_clean_250415_v2_with_blendshapes.pkl"
# full_path = path / filename
# df = pd.read_pickle(full_path)
# print(f"Loaded DataFrame with {len(df)} rows.")

# # Set data root path
# blendshapes_path = Path("/mnt/ML/Development/katya.ivantsiv/blendshapes")
# q_features_path = Path("/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")

# indices_to_drop = []

# for index, row in df.iterrows():
#     tar_id = row['tar_id']
#     run_path = row['run_path']
#     features_path = q_features_path / run_path / f'{tar_id}.npy'
    
#     landmarks_and_blendshapes_path = blendshapes_path / run_path / 'landmarks_and_blendshapes.npz'
#     landmarks_and_blendshapes = np.load(landmarks_and_blendshapes_path)
#     blendshape_label = landmarks_and_blendshapes['blendshapes']
    
#     # Check the shapes alignment of the blendshape_label and the features
#     bs_len = blendshape_label.shape[0]
    
#     features = np.load(features_path, mmap_mode='r')
#     features_len = features.shape[0]

#     if np.abs(features_len*30/200 - bs_len)>1:
#         indices_to_drop.append(index)
#         continue
    
#     # Check the landmarks
#     out_of_bounds_or_nan = np.any((blendshape_label < 0) | (blendshape_label > 1) | np.isnan(blendshape_label)) 
#     if out_of_bounds_or_nan:
#         indices_to_drop.append(index)

# df_cleaned = df.drop(index=indices_to_drop).reset_index(drop=True)

######################################################
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from collections import defaultdict

# # Load the original pickle
# print("Loading DataFrame from pickle...")
# path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
# filename = "LOUD_GIP_general_clean_250415_v2_with_blendshapes.pkl"
# full_path = path / filename
# df = pd.read_pickle(full_path)
# print(f"Loaded DataFrame with {len(df)} rows.")

# # Set data root path
# blendshapes_path = Path("/mnt/ML/Development/katya.ivantsiv/blendshapes")
# q_features_path = Path("/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")

# # run_path_to_tar_id2 = df.drop_duplicates(subset='run_path').set_index('run_path')['tar_id'].to_dict()
# run_path_to_tar_ids = df.groupby('run_path')[{'tar_id'}].unique().to_dict()

# indices_to_drop = []

# for run_path in run_path_to_tar_ids:
#     landmarks_and_blendshapes_path = blendshapes_path / run_path / 'landmarks_and_blendshapes.npz'
#     landmarks_and_blendshapes = np.load(landmarks_and_blendshapes_path)
#     blendshape_label = landmarks_and_blendshapes['blendshapes']
#     bs_len = blendshape_label.shape[0]
    
#     tar_ids = run_path_to_tar_ids[run_path]
#     for tar_id in tar_ids:
#         features_path = q_features_path / run_path / f'{tar_id}.npy'
#         features = np.load(features_path, mmap_mode='r')
#         features_len = features.shape[0]
#         # Check the shapes alignment of the blendshape_label and the features
#         if np.abs(features_len*30/200 - bs_len)>15:
#             indices_to_drop.append(index)
#             continue
    
#     # Check the landmarks
#     out_of_bounds_or_nan = np.any((blendshape_label < 0) | (blendshape_label > 1) | np.isnan(blendshape_label)) 
#     if out_of_bounds_or_nan:
#         indices_to_drop.append(index)



# df_cleaned = df.drop(index=indices_to_drop).reset_index(drop=True)