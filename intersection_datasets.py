import pandas as pd
from pathlib import Path

path = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits')
path1 = path / 'train_kfold_all_24p1M_with_side.pkl'
path2 = path / '4M_20250220_loud_valid_lmks.pkl'

df1 = pd.read_pickle(path1)
df2 = pd.read_pickle(path2)

# Ensure both have 'tar_id' as string
df1['tar_id'] = df1['tar_id'].astype(str)
df2['tar_id'] = df2['tar_id'].astype(str)

# Get intersections
common_tar_ids = set(df1['tar_id']) & set(df2['tar_id'])

print(f"Intersection count: {len(common_tar_ids)}")

# Optionally get intersecting rows
df1_common = df1[df1['tar_id'].isin(common_tar_ids)]
df2_common = df2[df2['tar_id'].isin(common_tar_ids)]

# Show first few entries
print(df1_common.head())

