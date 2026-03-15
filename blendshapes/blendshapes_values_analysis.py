from pathlib import Path
import pandas as pd
import numpy as np

blendshapes_path = Path("/mnt/ML/Development/katya.ivantsiv/blendshapes/")
split_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/loud_and_whisper_and_lip_20250713_064722.pkl")
df = pd.read_pickle(split_path)

for index, row in df.iterrows():
    npy_path = blendshapes_path / row.run_path / "blendshapes.npy"
    blendshape_label = np.load(npy_path)
    a=1
        