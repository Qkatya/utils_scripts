import pandas as pd
from pathlib import Path

# Input pickle files to merge
input_files = [
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_all.pkl"),
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_cheekPuff.pkl"),
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_eyeBlinkLeft.pkl"),
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_eyeBlinkRight.pkl"),
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_jawOpen.pkl"),
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_mouthPucker.pkl"),
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_mouthSmileLeft.pkl"),
    Path("/home/ido.kazma/projects/blendshape-fairseq/q-fairseq/tmp/valid_expressive_mouthSmileRight.pkl"),
]

output_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/expressive_validation.pkl")

# Load and concatenate all DataFrames
dfs = []
attrs = {}

for pkl_file in input_files:
    print(f"Loading {pkl_file.name}...")
    df = pd.read_pickle(pkl_file)
    dfs.append(df)
    # Merge attrs from each file
    attrs.update(getattr(df, 'attrs', {}))

# Combine all into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Remove duplicates based on tar_id if the column exists
if 'tar_id' in merged_df.columns:
    original_len = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset='tar_id', keep='first')
    if len(merged_df) < original_len:
        print(f"Removed {original_len - len(merged_df)} duplicate tar_ids")

# Preserve attrs
merged_df.attrs = attrs

# Save the result
merged_df.to_pickle(output_path)
print(f"\n✅ Merged {len(input_files)} files into: {output_path}")
print(f"   Total rows: {len(merged_df)}")



