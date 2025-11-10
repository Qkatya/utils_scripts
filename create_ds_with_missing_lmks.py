
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Load the original pickle
print("Loading DataFrame from pickle...")
path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
filename = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes_cleaned2_with_side.pkl"
full_path = path / filename
df = pd.read_pickle(full_path)

new_df = pd.read_pickle( path / '6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes_cleaned2_with_side_first_10_rows_missing_lmks_eith_attrs.pkl')
new_row = pd.DataFrame([{
    'tar_id': 'a700f763-8b1e-43ca-8fac-962a0669e8f8',
    'instruction_type': 'loud',
    'read_text': 'bla bla',
    'run_path': '2024/07/01/QaQa-152527/10_0_ab38ed21-af51-49c3-806d-192658243be0_loud',
    'frame_num': 1216,
    'side': 'left'
}])

new_df = pd.concat([new_df, df.iloc[222:228]], ignore_index=True)
new_df.attrs = df.attrs

new_filename = filename.replace(".pkl", "_missing_lmks.pkl")
new_df.to_pickle(path / new_filename)

a=1

######


# Set the path to the directory containing .pkl files
input_dir = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/existing_files')

# Get all .pkl files in the directory
pkl_files = sorted(input_dir.glob('*.pkl'))

# Load and concatenate all DataFrames
dfs = []
for pkl_file in pkl_files:
    print(f"Loading {pkl_file.name}...")
    df = pd.read_pickle(pkl_file)
    dfs.append(df)

# Combine all into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)


merged_df.attrs = df.attrs


# Save the result
output_path = path / '4M_20250220_loud_lmks.pkl'
merged_df.to_pickle(output_path)
print(f"\n✅ Merged {len(pkl_files)} files into: {output_path}")