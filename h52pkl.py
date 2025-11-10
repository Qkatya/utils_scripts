import pandas as pd
import h5py
from pathlib import Path

# Paths
out_dir = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits')
base_dir = Path('/mnt/ML/Development/ML_Data_DB/v2/splits/full_hdf5/20250527_split_1')
file_name = 'train_kfold_all_24p1M.h5'
h5_path = base_dir / file_name
out_path = out_dir / file_name
output_pkl_path = out_path.with_suffix(".pkl")

print(f"Loading {h5_path.name}...")

# Load the DataFrame
df = pd.read_hdf(h5_path, key=h5_path.stem)

print(f"Finished Loading {h5_path.name} with {len(df)} rows")


# Load the extra dataset and attributes from h5py
with h5py.File(h5_path, 'r') as h5file:
    sizes = list(h5file['sizes'])  # Convert to list of ints
    features_path = h5file.attrs['features_path']
    hubert_soft_path = h5file.attrs['hubert_soft_path']
    hubert_asr_path = h5file.attrs['hubert_asr_path']

# Optionally restore sizes as a column
df['frame_num'] = sizes

# Save as pickle
df.to_pickle(output_pkl_path)
print(f"Saved DataFrame to {output_pkl_path}")

# Optional: print metadata
print("Loaded attributes:")
print(" - features_path:", features_path)
print(" - hubert_soft_path:", hubert_soft_path)
print(" - hubert_asr_path:", hubert_asr_path)
