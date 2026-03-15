from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict
from handle_split import get_side_from_db_parallel, transfer_attrs
from itertools import islice

filename = "/mnt/ML/Personalized/ana.polterovich/tar_id_instruct_type_dict.pkl"
print('load pickl')
tar_id_instruct_type_dict = pd.read_pickle(filename)

# sampled_dict = dict(islice(tar_id_instruct_type_dict.items(), 100))
# del tar_id_instruct_type_dict
# tar_id_instruct_type_dict = sampled_dict

print('get tar_ids with whisper labels')
label_to_keys = defaultdict(list)
for k, v in tqdm(tar_id_instruct_type_dict.items()):
    # if v in ['whisper']: # 'lip', 
    if v in ['lip']: # 'lip', 
        label_to_keys[v].append(k)
# filtered_tarids = label_to_keys['whisper']
filtered_tarids = label_to_keys['lip']

print(f'There are {len(filtered_tarids)} filtered samples')

kfold_df_path = Path('/mnt/ML/Development/ML_Data_DB/v2/splits/full/20250616_split_1/') / 'train_kfold_18p4M.pkl'
kfold_df = pd.read_pickle(kfold_df_path)

# Get only whisper samples from the kfold splits
whisper_kfold_samples = kfold_df[kfold_df['tar_id'].isin(filtered_tarids)]

# Add side column from db and attributes from other split
whisper_kfold_samples_with_side = get_side_from_db_parallel(whisper_kfold_samples, n_jobs=10)
whisper_kfold_samples_with_side_with_attrs = transfer_attrs(whisper_kfold_samples_with_side)

save_path = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits') /'train_kfold_18p4M_lip.pkl'
whisper_kfold_samples_with_side.to_pickle(save_path)
print(f'df with size of {len(whisper_kfold_samples_with_side)} was saved to {save_path}')



