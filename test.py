from pathlib import Path
import pandas as pd

task_data=Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
dataset_train_subset=Path("loud_and_whisper_20250709_202415.pkl")

df = pd.read_pickle(task_data / dataset_train_subset)
a=1