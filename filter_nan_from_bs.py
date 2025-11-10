from pathlib import Path
import pandas as pd

# Load the original pickle
print("Loading DataFrame from pickle...")
path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
filename = "LOUD_GIP_general_clean_250415_v2.pkl"
# filename_new = "LOUD_GIP_general_clean_250415_v2_with_blendshapes.pkl"
full_path = path / filename
df = pd.read_pickle(full_path)
print(f"Loaded DataFrame with {len(df)} rows.")

# Set data root path
data_path = Path("/mnt/A3000/Recordings/v2_data")

# Start checking for .npz file existence
print("Checking for existing 'landmarks_and_blendshapes.npz' files...")

# Apply a lambda to construct full paths and check existence
full_paths = df['run_path'].apply(lambda p: data_path / p / "landmarks_and_blendshapes.npz")

# Add progress print every N rows checked
exists_mask = []
for i, p in enumerate(full_paths):
    exists_mask.append(p.exists())
    if i % 500 == 0 or i == len(full_paths) - 1:
        print(f"Checked {i+1}/{len(full_paths)} paths...")

# Filter the DataFrame
npz_df = df[exists_mask].copy()
print(f"Found {len(npz_df)} rows with .npz files.")

# Show a preview of the result
print("Preview of filtered DataFrame:")
print(npz_df.head())

# Save filtered DataFrame
output_path = "/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes.pkl"
print(f"Saving filtered DataFrame to: {output_path}")
npz_df.to_pickle(output_path)
print("Save complete.")

a = 1

