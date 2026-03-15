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
shrinked_filename = filename.replace(".pkl", "_first_100_rows.pkl")
df.head(100).to_pickle(path / shrinked_filename)
print(f"Saved cleaned DataFrame with {len(shrinked_filename)} rows to {shrinked_filename}")
