import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pandas as pd
import h5py
from pathlib import Path

# Read the h5 file
input_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/blendshapes_KatyaIvantsiv_2026_01_11_with_side_20260112.h5")
output_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/blendshapes_KatyaIvantsiv_2026_01_11_with_side_20260112_x300.h5")
good_file_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/expressive_validation_with_side_20260111.h5")

# Load DataFrame
train_df = pd.read_hdf(input_path)

# Load sizes from input file
with h5py.File(input_path, 'r') as h5file:
    sizes = list(h5file['sizes'][:])

# Load attributes from the good file
with h5py.File(good_file_path, 'r') as h5file:
    features_path = h5file.attrs.get('features_path', '')
    hubert_soft_path = h5file.attrs.get('hubert_soft_path', '')
    hubert_asr_path = h5file.attrs.get('hubert_asr_path', '')

print(f"Original rows: {len(train_df)}")
print(f"Original sizes: {len(sizes)}")

# Replicate rows and sizes 300 times
train_df = pd.concat([train_df] * 300).reset_index(drop=True)
sizes = sizes * 300

print(f"New rows: {len(train_df)}")
print(f"New sizes: {len(sizes)}")

# Save to new h5 file with proper structure
key_name = output_path.stem

with pd.HDFStore(output_path, mode='w', swmr=True) as store:
    store.put(key_name, train_df, format='table', complevel=9, complib='blosc')
    store.flush(fsync=True)

with h5py.File(output_path, 'a') as h5file:
    h5file.create_dataset('sizes', data=sizes)
    if features_path:
        h5file.attrs['features_path'] = features_path
    if hubert_soft_path:
        h5file.attrs['hubert_soft_path'] = hubert_soft_path
    if hubert_asr_path:
        h5file.attrs['hubert_asr_path'] = hubert_asr_path
    
    print(f"Keys: {list(h5file.keys())}")
    print(f"features_path: {h5file.attrs.get('features_path', 'N/A')}")

print(f"Saved to: {output_path}")