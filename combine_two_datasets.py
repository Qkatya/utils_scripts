import pandas as pd
from pathlib import Path
from datetime import datetime

# Define paths
path1 = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/loud_and_whisper_20250709_202415.pkl")
path2 = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/snipets/train_kfold_18p4M_lip_valid_20250710_062327.pkl")
output_dir = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")

# Load DataFrames
df1 = pd.read_pickle(path1)
df2 = pd.read_pickle(path2)

# Combine
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset='tar_id', keep='first')  # or 'last'

# Optionally combine .attrs dictionaries (if present)
df_combined.attrs = {**getattr(df1, 'attrs', {}), **getattr(df2, 'attrs', {})}

# Create output filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"loud_and_whisper_and_lip_{timestamp}.pkl"

# Save combined DataFrame
df_combined.to_pickle(output_path)

print(f"✅ Combined DataFrame saved to: {output_path}")
