import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
from pathlib import Path

def get_df_with_side(df_with_tar_id, batch_size=100000):
    assert df_with_tar_id.tar_id.value_counts().max()==1
    assert isinstance(df_with_tar_id.tar_id.iloc[0], str)
    
    # DB setup
    engine = create_engine("postgresql://bits_viewer:qviewer1@q-data-db-replica.q.ai:5432/q-data-db-prod")

    results = []
    
    tar_ids = df_with_tar_id.tar_id
    
    for i in tqdm(range(0, len(tar_ids), batch_size)):
        batch_ids = tar_ids[i:i + batch_size]
        placeholders = ",".join([f"'{x}'" for x in batch_ids])
        #dwh.q_frames_details
        query = """
        select id as tar_id, side
        from q_frames_files
        WHERE id IN ({})
        """.format(placeholders)
        
        # df_batch = pd.read_sql(query, engine,params batch_ids)
        df_batch = pd.read_sql(query, engine)
        df_batch["tar_id"] = df_batch["tar_id"].astype(str)
        
        results.append(df_batch)
    
    # Combine all batches
    df_all_results = pd.concat(results, ignore_index=True)
    
    # Merge back into original df
    df_with_tar_id = df_with_tar_id.merge(df_all_results, how='left', on='tar_id')

    return df_with_tar_id

out_dir = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits')
base_dir = Path('/mnt/ML/Development/ML_Data_DB/v2/splits/full/20250708_split_1')
# file_name = 'train_kfold_all_24p1M.pkl'
files=[
  "train_kfold_19p8M"]
for file_name in files:
    path = (base_dir / file_name).with_suffix(".pkl")
    print('loading pkl')
    df = pd.read_pickle(path)
    print(f'finished loading pkl with {len(df)} rows')
    df_with_side = get_df_with_side(df, batch_size=100000)
    out_path = out_dir/ file_name
    new_path = out_path.with_name(out_path.stem + '_20250708_split_1_with_side' + out_path.suffix)
    df_with_side.to_pickle(new_path)
a=1