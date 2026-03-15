import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from pathlib import Path
import h5py
import numpy as np

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
        with engine.connect() as conn:
            df_batch = pd.read_sql_query(query, conn.connection)
        df_batch["tar_id"] = df_batch["tar_id"].astype(str)
        
        results.append(df_batch)
    
    # Combine all batches
    df_all_results = pd.concat(results, ignore_index=True)
    
    # Merge back into original df
    df_with_tar_id = df_with_tar_id.merge(df_all_results, how='left', on='tar_id')

    return df_with_tar_id

out_dir = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits')
base_dir = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits')
files=[
  "train_filtered_merged_p5.0.pkl"]
for file_name in files:
    path = base_dir / file_name
    print(f'loading {path.suffix} file')
    
    # Load file based on extension
    if path.suffix in ['.h5', '.hdf5']:
        df = pd.read_hdf(path)
        # Load sizes and attributes from h5py
        with h5py.File(path, 'r') as f:
            sizes = np.array(f['sizes'][:], dtype=np.int64)
            attrs = {
                'features_path': f.attrs.get('features_path', ''),
                'hubert_soft_path': f.attrs.get('hubert_soft_path', ''),
                'hubert_asr_path': f.attrs.get('hubert_asr_path', '')
            }
    else:
        df = pd.read_pickle(path)
    
    print(f'finished loading file with {len(df)} rows')
    df_with_side = get_df_with_side(df, batch_size=100000)
    out_path = out_dir / file_name
    new_path = out_path.with_name(out_path.stem + '_with_side' + out_path.suffix)
    
    # Save in same format as input
    if path.suffix in ['.h5', '.hdf5']:
        # First: create h5py file with sizes and attributes
        with h5py.File(new_path, 'w') as h5file:
            h5file.create_dataset('sizes', data=sizes)
            for k, v in attrs.items():
                if v:  # Only set if not empty
                    h5file.attrs[k] = v
        # Then: append DataFrame using HDFStore
        with pd.HDFStore(new_path, mode='a', swmr=True) as store:
            store.put(new_path.stem, df_with_side, format='table', complevel=9, complib='blosc')
            store.flush(fsync=True)
        print(f'Saved h5: {new_path}')
    else:
        df_with_side.to_pickle(new_path)
        print(f'Saved pickle: {new_path}')